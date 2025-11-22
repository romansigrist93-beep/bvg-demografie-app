import streamlit as st
import requests
import pandas as pd
import math

# ======================================================================
# 0. STREAMLIT KONFIG
# ======================================================================

st.set_page_config(
    layout="wide",
    page_title="BVG-Umwandlungssatz-Analyse",
)

# Ab welchem Alter wird im Modell mit Sparen begonnen?
SAVING_START_AGE = 25

# ======================================================================
# 1. MATHE-KERNFUNKTIONEN
# ======================================================================

def annuity_factor(years: float, rate: float) -> float:
    """Barwertfaktor einer jährlichen Rente 1 (nachschüssig)."""
    if rate <= 0:
        return years
    return (1 - (1 + rate) ** (-years)) / rate


def fair_conversion_rate(life_expectancy_years: float, discount_rate: float) -> float:
    """Theoretisch fairer UWS (dezimal)."""
    a = annuity_factor(life_expectancy_years, discount_rate)
    if a == 0:
        return 0.0
    return 1 / a


def adjust_factor_from_burden(
    burden_index: float,
    ref_burden: float = 0.5,
    elasticity: float = 1.0,
) -> float:
    """
    Anpassungsfaktor für die Rendite in Abhängigkeit des Belastungsindex.
    B > B_ref  -> Faktor < 1  (Rendite runter)
    B < B_ref  -> Faktor > 1  (Rendite rauf)
    """
    delta = burden_index - ref_burden
    factor = 1 - elasticity * delta
    factor = max(0.2, min(1.2, factor))  # nicht völlig eskalieren lassen
    return factor


def compute_capital_at_retirement(
    salary: float,
    contrib_rate: float,
    years_to_retirement: int,
    return_rate: float,
    contrib_factor: float,
) -> float:
    """
    Kapitalaufbau (Endwert einer nachschüssigen Rente).
    Vereinfachung: BVG-Lohn = Jahreslohn, kein Koordinationsabzug.
    """
    r = return_rate
    cf = contrib_factor
    annual_contrib = salary * contrib_rate * cf

    if years_to_retirement <= 0:
        return 0.0

    if r <= 0:
        capital = annual_contrib * years_to_retirement
    else:
        future_value_factor = ((1 + r) ** years_to_retirement - 1) / r
        capital = annual_contrib * future_value_factor

    return capital


def compute_bvg_salary(gross_salary: float, coord_deduction: float) -> float:
    """BVG-pflichtiger Lohn = max(0, Bruttolohn - Koordinationsabzug)."""
    return max(0.0, gross_salary - coord_deduction)


# ======================================================================
# 2. SZENARIEN & PARAMETER
# ======================================================================

@st.cache_data
def load_scenario_metadata():
    """
    Liest BFS-Metadaten und baut eine Heuristik für Rendite/ Diskont je Szenario.
    """
    url_pop_meta = (
        "https://www.pxweb.bfs.admin.ch/api/v1/de/px-x-0104000000_102/"
        "px-x-0104000000_102.px?metadata=true"
    )
    try:
        meta_pop = requests.get(url_pop_meta).json()
        vars_list = meta_pop.get("variables", [])
        scen_var = next(
            (v for v in vars_list if v.get("code") == "Szenario-Variante"),
            None,
        )
        if scen_var:
            codes = scen_var["values"]
            labels = scen_var["valueTexts"]
            scen_labels = {str(code): label for code, label in zip(codes, labels)}
        else:
            scen_labels = {}
    except Exception:
        scen_labels = {}

    scenario_params = {
        "0": {
            "name": scen_labels.get("0", "Referenzszenario A-00-2025"),
            "return_active": 0.025,
            "discount_retired": 0.015,
            "contrib_factor": 1.00,
        },
        "1": {
            "name": scen_labels.get("1", "'hohes' Szenario B-00-2025"),
            "return_active": 0.030,
            "discount_retired": 0.020,
            "contrib_factor": 1.05,
        },
        "2": {
            "name": scen_labels.get("2", "'tiefes' Szenario C-00-2025"),
            "return_active": 0.020,
            "discount_retired": 0.010,
            "contrib_factor": 0.95,
        },
        "9": {
            "name": scen_labels.get("9", "Variante A-05-2025 'hohes Wanderungssaldo'"),
            "return_active": 0.030,
            "discount_retired": 0.018,
            "contrib_factor": 1.05,
        },
        "10": {
            "name": scen_labels.get("10", "Variante A-06-2025 'tiefes Wanderungssaldo'"),
            "return_active": 0.022,
            "discount_retired": 0.012,
            "contrib_factor": 0.98,
        },
        "11": {
            "name": scen_labels.get("11", "Variante A-07-2025 'stabile Wanderungsbewegungen'"),
            "return_active": 0.025,
            "discount_retired": 0.015,
            "contrib_factor": 1.00,
        },
        "12": {
            "name": scen_labels.get("12", "Variante A-08-2025 'Stark verbesserte Vereinbarkeit'"),
            "return_active": 0.020,
            "discount_retired": 0.010,
            "contrib_factor": 0.90,
        },
        "14": {
            "name": scen_labels.get("14", "Variante A-10-2025 'Stärkere Erwerbsbeteiligung im höheren Alter'"),
            "return_active": 0.030,
            "discount_retired": 0.020,
            "contrib_factor": 1.08,
        },
        "15": {
            "name": scen_labels.get("15", "Variante A-11-2025 'Geringere Erwerbsbeteiligung im höheren Alter'"),
            "return_active": 0.021,
            "discount_retired": 0.011,
            "contrib_factor": 0.93,
        },
    }
    return scenario_params


# ======================================================================
# 3. BFS: LEBENSERWARTUNG (ex) FÜR ALTER 60–70
# ======================================================================

@st.cache_data
def load_le_data():
    """
    Laedt die Lebenserwartung (ex) vom BFS für Alter 60–70, Jahre 2025–2055.
    DataFrame-Spalten: Jahr, Geschlecht, Alter (Code), AgeNum, LifeExpectancy
    """
    url_le = "https://www.pxweb.bfs.admin.ch/api/v1/de/px-x-0102020300_102/px-x-0102020300_102.px"
    url_meta_le = url_le + "?metadata=true"

    meta_le = requests.get(url_meta_le).json()
    vars_le = meta_le["variables"]

    def get_var_le(code: str):
        return next(v for v in vars_le if v.get("code") == code)

    var_year = get_var_le("Jahr")
    var_sex = get_var_le("Geschlecht")
    var_age = get_var_le("Alter")
    var_unit = get_var_le("Beobachtungseinheit")

    # Jahr-Codes 2025–2055
    year_codes = []
    year_map = {}
    for code, txt in zip(var_year["values"], var_year["valueTexts"]):
        y = int(txt)
        if 2025 <= y <= 2055:
            year_codes.append(code)
            year_map[code] = y

    # Geschlechter
    sex_codes = var_sex["values"]
    sex_map = dict(zip(var_sex["values"], var_sex["valueTexts"]))

    # Alterscodes für 60–70 Jahre
    age_values = var_age["values"]
    age_texts = var_age["valueTexts"]
    age_map_text = dict(zip(age_values, age_texts))

    age_codes = []
    for code, txt in age_map_text.items():
        nums = "".join(ch for ch in txt if ch.isdigit())
        if nums:
            age_num = int(nums)
            if 60 <= age_num <= 70:
                age_codes.append(code)

    # Beobachtungseinheit: Lebenserwartung (ex)
    unit_code_ex = next(
        code
        for code, txt in zip(var_unit["values"], var_unit["valueTexts"])
        if txt.startswith("Lebenserwartung (ex)")
    )

    body_le = {
        "query": [
            {"code": "Jahr", "selection": {"filter": "item", "values": year_codes}},
            {"code": "Geschlecht", "selection": {"filter": "item", "values": sex_codes}},
            {"code": "Alter", "selection": {"filter": "item", "values": age_codes}},
            {
                "code": "Beobachtungseinheit",
                "selection": {"filter": "item", "values": [unit_code_ex]},
            },
        ],
        "response": {"format": "JSON"},
    }

    r_le = requests.post(url_le, json=body_le)
    r_le.raise_for_status()
    j_le = r_le.json()

    rows = j_le["data"]
    keys = [row["key"] for row in rows]
    values = [float(row["values"][0]) for row in rows]

    col_names = [c["code"] for c in j_le["columns"][:4]]  # Jahr, Geschlecht, Alter, Einheit
    le_df = pd.DataFrame(keys, columns=col_names)
    le_df["LifeExpectancy"] = values

    # Map Jahr & Geschlecht
    le_df["Jahr"] = le_df["Jahr"].map(year_map)
    le_df["Geschlecht"] = le_df["Geschlecht"].map(sex_map)

    # AgeNum extrahieren
    def parse_age(code: str) -> int:
        txt = age_map_text[code]
        nums = "".join(ch for ch in txt if ch.isdigit())
        return int(nums) if nums else 0

    le_df["AgeNum"] = le_df["Alter"].map(parse_age)

    return le_df, sex_map


def get_life_expectancy(year: int, sex_label: str, age_ret: int, le_df: pd.DataFrame) -> float:
    """
    Liefert die Lebenserwartung ex ab dem gewählten Pensionierungsalter.
    Year und Age werden, falls nötig, in den verfügbaren Bereich geclamped.
    """
    years_all = le_df["Jahr"].unique()
    ages_all = le_df["AgeNum"].unique()

    year_clamped = int(min(max(year, years_all.min()), years_all.max()))
    age_clamped = int(min(max(age_ret, ages_all.min()), ages_all.max()))

    sub = le_df[
        (le_df["Jahr"] == year_clamped)
        & (le_df["Geschlecht"] == sex_label)
        & (le_df["AgeNum"] == age_clamped)
    ]
    if sub.empty:
        raise ValueError(
            f"Keine Lebenserwartung fuer Jahr={year_clamped}, Alter={age_clamped}, "
            f"Geschlecht={sex_label} gefunden."
        )
    return float(sub["LifeExpectancy"].iloc[0]), year_clamped, age_clamped


# ======================================================================
# 4. BFS: BEVÖLKERUNGSSZENARIEN → BELASTUNGSINDEX
# ======================================================================

@st.cache_data
def load_burden_from_bfs():
    """
    Holt die BFS-Bevölkerungsszenarien und berechnet für jedes
    Szenario & Jahr den Belastungsindex = 65+ / 20–64.

    Tabelle: px-x-0104000000_102
    """
    url = "https://www.pxweb.bfs.admin.ch/api/v1/de/px-x-0104000000_102/px-x-0104000000_102.px"
    url_meta = url + "?metadata=true"

    meta = requests.get(url_meta).json()
    vars_ = meta["variables"]

    def gv(code):
        return next(v for v in vars_ if v.get("code") == code)

    var_year = gv("Jahr")
    var_scen = gv("Szenario-Variante")
    var_age = gv("Alter")
    var_sex = gv("Geschlecht")
    var_nat = gv("Staatsangehörigkeit (Kategorie)")
    var_unit = gv("Beobachtungseinheit")

    # Jahre 2025–2055
    year_codes = []
    year_map = {}
    for code, txt in zip(var_year["values"], var_year["valueTexts"]):
        y = int(txt)
        if 2025 <= y <= 2055:
            year_codes.append(code)
            year_map[code] = y

    scen_codes = var_scen["values"]
    age_codes = var_age["values"]
    age_map_text = dict(zip(var_age["values"], var_age["valueTexts"]))

    # Geschlecht Total
    try:
        sex_total = next(
            c
            for c, txt in zip(var_sex["values"], var_sex["valueTexts"])
            if "Total" in txt
        )
    except StopIteration:
        sex_total = var_sex["values"][0]

    # Staatsangehörigkeit Total
    try:
        nat_total = next(
            c
            for c, txt in zip(var_nat["values"], var_nat["valueTexts"])
            if "Total" in txt
        )
    except StopIteration:
        nat_total = var_nat["values"][0]

    # Einheit: Bevölkerungsstand / -bestand
    unit_pop = next(
        c
        for c, txt in zip(var_unit["values"], var_unit["valueTexts"])
        if "Bevölkerungsstand" in txt or "Bevölkerungsbestand" in txt
    )

    body = {
        "query": [
            {
                "code": "Szenario-Variante",
                "selection": {"filter": "item", "values": scen_codes},
            },
            {"code": "Jahr", "selection": {"filter": "item", "values": year_codes}},
            {"code": "Alter", "selection": {"filter": "item", "values": age_codes}},
            {
                "code": "Geschlecht",
                "selection": {"filter": "item", "values": [sex_total]},
            },
            {
                "code": "Staatsangehörigkeit (Kategorie)",
                "selection": {"filter": "item", "values": [nat_total]},
            },
            {
                "code": "Beobachtungseinheit",
                "selection": {"filter": "item", "values": [unit_pop]},
            },
        ],
        "response": {"format": "JSON"},
    }

    r = requests.post(url, json=body)
    r.raise_for_status()
    j = r.json()

    rows = j["data"]
    keys = [row["key"] for row in rows]
    vals = [float(row["values"][0]) for row in rows]

    col_names = [c["code"] for c in j["columns"][:-1]]
    df = pd.DataFrame(keys, columns=col_names)
    df["value"] = vals

    # Codes → Jahr
    df["Jahr"] = df["Jahr"].map(year_map)

    def parse_age(code):
        txt = age_map_text[code]  # z.B. "65 Jahre", "100 Jahre und mehr"
        nums = "".join(ch for ch in txt if ch.isdigit())
        return int(nums) if nums else 0

    df["AgeNum"] = df["Alter"].map(parse_age)

    # 20–64 = Erwerbstätige, 65+ = Pensionierte
    df_work = df[(df["AgeNum"] >= 20) & (df["AgeNum"] <= 64)]
    df_old = df[df["AgeNum"] >= 65]

    work = (
        df_work.groupby(["Szenario-Variante", "Jahr"])["value"]
        .sum()
        .rename("WorkPop")
    )
    old = (
        df_old.groupby(["Szenario-Variante", "Jahr"])["value"]
        .sum()
        .rename("OldPop")
    )

    burden_df = pd.concat([work, old], axis=1).reset_index()
    burden_df["BurdenIndex"] = burden_df["OldPop"] / burden_df["WorkPop"]

    return burden_df


def get_burden_index_from_df(
    scen_code: str,
    year: int,
    burden_df: pd.DataFrame,
) -> float:
    """Hilfsfunktion: nimmt das nächstliegende Jahr im verfügbaren Bereich."""
    years_all = burden_df["Jahr"].unique()
    year_clamped = int(min(max(year, years_all.min()), years_all.max()))

    sub = burden_df[
        (burden_df["Szenario-Variante"] == scen_code)
        & (burden_df["Jahr"] == year_clamped)
    ]
    if sub.empty:
        raise ValueError(
            f"Kein Belastungsindex fuer Szenario {scen_code}, Jahr {year_clamped}"
        )
    return float(sub["BurdenIndex"].iloc[0])


# ======================================================================
# 5. STREAMLIT-APP
# ======================================================================

def main():
    # ruhige Akzentfarbe
    st.markdown(
        """
        <style>
        h1, h2, h3, h4 {
            color: #00857C !important;
            font-family: "Segoe UI", sans-serif;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### ZHAW – CAS Datenkompetenz in Versicherungen")
    st.title("Pensionskassen-Modellierung: UWS, Lebenserwartung & demografische Belastung")

    # Daten laden
    le_df, sex_map = load_le_data()
    scenario_params = load_scenario_metadata()
    burden_df = load_burden_from_bfs()

    scen_options = {p["name"]: code for code, p in scenario_params.items()}

    tab_model, tab_burden, tab_compare, tab_method = st.tabs(
        ["🔢 Modell", "📊 Belastungsindex & Rendite", "📊 Szenario-Impact", "📐 Methodik"]
    )

    # ------------------------------------------------------------------
    # TAB 1: MODELL
    # ------------------------------------------------------------------
    with tab_model:
        st.subheader("1. Eingaben ↩️")

        col1, col2, col3 = st.columns(3)

        # --- Spalte 1: Szenario ---
        with col1:
            scen_name = st.selectbox(
                "Ökonomisches/demografisches Szenario",
                options=list(scen_options.keys()),
                index=0,
            )
            scen_code = scen_options[scen_name]
            params = scenario_params[scen_code]

        # --- Spalte 2: Geschlecht & Alter ---
        with col2:
            sex_label = st.selectbox(
                "Geschlecht",
                options=list(sex_map.values()),
                index=1 if any("Frau" in v for v in sex_map.values()) else 0,
            )
            age_now = st.slider("Aktuelles Alter", 20, 64, 35)
            ret_age = st.slider("Pensionierungsalter", 60, 70, 65)

        # Sparjahre ab heute
        years_to_ret = ret_age - age_now
        if years_to_ret <= 0:
            st.error("Das Pensionierungsalter muss grösser sein als das aktuelle Alter.")
            return

        # Sparjahre insgesamt (ab 25) und bereits vergangene Jahre
        years_past = max(0, age_now - SAVING_START_AGE)
        years_total = max(0, ret_age - SAVING_START_AGE)

        # --- Spalte 3: Startjahr, Rentenjahr wird daraus berechnet ---
        min_year = int(le_df["Jahr"].min())
        max_year = int(le_df["Jahr"].max())
        max_start_for_ages = max_year - years_to_ret

        with col3:
            if max_start_for_ages <= min_year:
                # kein sinnvoller Slider möglich -> Startjahr fix setzen
                start_year = min_year
                st.markdown(f"**Startjahr (fix): {start_year}**")
                st.caption(
                    "Der Projektionshorizont ist ausgeschöpft – "
                    "das Startjahr wird automatisch auf das erste verfügbare Jahr gesetzt."
                )
            else:
                start_year = st.slider(
                    "Startjahr (heute im Modell)",
                    min_value=min_year,
                    max_value=max_start_for_ages,
                    value=min_year,
                    step=1,
                )

            year_ret = start_year + years_to_ret
            st.markdown(f"**Automatisch berechnetes Rentenjahr:** {year_ret}")

        st.subheader("2. Finanzielle Parameter")

        col_fin1, col_fin2 = st.columns(2)

        with col_fin1:
            salary = st.number_input(
                "Jährlicher Bruttolohn in CHF",
                min_value=40_000,
                max_value=200_000,
                value=80_000,
                step=5_000,
            )
            coord_deduction = st.number_input(
                "Koordinationsabzug in CHF",
                min_value=0,
                max_value=50_000,
                value=25_725,
                step=500,
            )
            contrib = st.slider(
                "BVG-Beitragssatz",
                min_value=0.05,
                max_value=0.20,
                value=0.12,
                step=0.01,
                format="%.2f",
            )

        with col_fin2:
            uws_chosen = (
                st.slider(
                    "Gewählter / regulatorischer UWS in %",
                    min_value=3.0,
                    max_value=7.0,
                    value=5.0,
                    step=0.1,
                )
                / 100.0
            )
            elasticity = st.slider(
                "Sensitivität der Rendite auf demografische Belastung",
                min_value=0.0,
                max_value=2.0,
                value=1.0,
                step=0.1,
            )

        # BVG-pflichtiger Lohn
        insured_salary = compute_bvg_salary(salary, coord_deduction)

        # Lebenserwartung: falls Rentenjahr > letzter BFS-Wert, letztes Jahr verwenden
        max_le_year = int(le_df["Jahr"].max())
        year_for_le_raw = year_ret
        year_for_le = min(year_for_le_raw, max_le_year)

        try:
            le, year_used, age_used = get_life_expectancy(
                year_for_le, sex_label, ret_age, le_df
            )
        except ValueError as e:
            st.error(str(e))
            return

        # Hinweise bei Clamping
        if year_used < year_ret:
            st.info(
                f"Für das Rentenjahr {year_ret} liegen keine BFS-Lebenserwartungen vor. "
                f"Es wird das letzte verfügbare Jahr {year_used} verwendet."
            )
        if age_used != ret_age:
            st.info(
                f"Für das Pensionierungsalter {ret_age} liegt keine direkte Lebenserwartung vor. "
                f"Es wird das nächste verfügbare Alter {age_used} verwendet."
            )

        # Belastungsindex während der verbleibenden Sparphase (ab heute)
        burden_list = []
        factor_list = []
        year_list = []

        for i in range(years_to_ret):
            y = start_year + i
            b = get_burden_index_from_df(scen_code, y, burden_df)
            burden_list.append(b)
            f = adjust_factor_from_burden(b, ref_burden=0.5, elasticity=elasticity)
            factor_list.append(f)
            year_list.append(y)

        avg_factor = sum(factor_list) / len(factor_list) if factor_list else 1.0

        # effektive Rendite und Kapital (gesamte Sparphase ab 25)
        r_base = params["return_active"]
        r_effective = r_base * avg_factor

        capital = compute_capital_at_retirement(
            salary=insured_salary,
            contrib_rate=contrib,
            years_to_retirement=years_total,
            return_rate=r_effective,
            contrib_factor=params["contrib_factor"],
        )

        # fairer UWS & Renten (ab Pensionierungsalter)
        uws_fair = fair_conversion_rate(
            life_expectancy_years=le,
            discount_rate=params["discount_retired"],
        )

        pension_chosen = capital * uws_chosen
        pension_fair = capital * uws_fair
        delta = pension_chosen - pension_fair

        st.markdown("---")
        st.subheader("3. Resultate")

        col_res1, col_res2, col_res3 = st.columns(3)

        col_res1.metric(
            label="Alterskapital (2. Säule)",
            value=f"CHF {capital:,.0f}",
            delta=f"Total Sparjahre: {years_total} (davon {years_past} bereits, {years_to_ret} zukünftig)",
        )

        col_res2.metric(
            label=f"Lebenserwartung ab Pensionierungsalter {age_used} "
                  f"(BFS-Jahr {year_used})",
            value=f"{le:.2f} Jahre",
            delta=f"Techn. Zins (d): {params['discount_retired']*100:.2f} %",
        )

        col_res3.metric(
            label="Fairer Umwandlungssatz (UWS)",
            value=f"{uws_fair*100:.2f} %",
            delta="Rein versicherungsmathematisch",
        )

        st.markdown("### Rentenvergleich")

        col_r1, col_r2, col_r3 = st.columns(3)

        col_r1.metric(
            label=f"Jahresrente (gewählter UWS: {uws_chosen*100:.2f} %)",
            value=f"CHF {pension_chosen:,.0f}",
        )

        col_r2.metric(
            label=f"Faire Jahresrente (UWS: {uws_fair*100:.2f} %)",
            value=f"CHF {pension_fair:,.0f}",
        )

        delta_text = f"CHF {abs(delta):,.0f} pro Jahr"
        if abs(delta) < 1:
            col_r3.metric(
                label="Über-/Unterdeckung",
                value="Praktisch neutral",
                delta="≈ CHF 0",
            )
        elif delta > 0:
            col_r3.metric(
                label="Über-/Unterdeckung",
                value="Überdeckung",
                delta=delta_text,
                delta_color="normal",
            )
        else:
            col_r3.metric(
                label="Über-/Unterdeckung",
                value="Unterdeckung",
                delta=delta_text,
                delta_color="inverse",
            )

        st.markdown("### Lohnbasis (Info)")
        st.write(
            f"- Bruttolohn: **CHF {salary:,.0f}**  \n"
            f"- Koordinationsabzug: **CHF {coord_deduction:,.0f}**  \n"
            f"- BVG-pflichtiger Lohn: **CHF {insured_salary:,.0f}**"
        )

        st.markdown("### Visualisierung der Jahresrenten")
        chart_df = pd.DataFrame(
            {
                "Variante": ["Gewählt", "Fair"],
                "Jahresrente": [pension_chosen, pension_fair],
            }
        )
        st.bar_chart(chart_df.set_index("Variante"))

    # ------------------------------------------------------------------
    # TAB 2: BELASTUNGSINDEX
    # ------------------------------------------------------------------
    with tab_burden:
        st.subheader("Belastungsindex & Renditepfad in der verbleibenden Sparphase")

        st.markdown(
            """
            **Belastungsindex** = Bevölkerung 65+ geteilt durch Bevölkerung 20–64  
            (berechnet direkt aus den BFS-Bevölkerungsszenarien).

            Die effektive Rendite in der Sparphase wird im Modell **zweistufig** bestimmt:

            1. Jedes Szenario hat einen eigenen *Basiszins* (Kapitalmarkt-Umfeld).  
            2. Dieser Basiszins wird über die Zeit mit einem Faktor aus dem Belastungsindex
               multipliziert. Szenarien mit höherer demografischer Belastung führen damit
               im Durchschnitt zu einer tieferen effektiven Rendite.

            Wichtig: Die Szenariobezeichnung (z.B. „niedrige Geburtenhäufigkeit“) ist
            eine **BFS-Kurzbeschreibung**. Entscheidend für die Rendite im Modell ist
            der tatsächlich berechnete Belastungsindex-Pfad in deinem Sparfenster –
            nicht das Label an sich.
            """
        )

        if "years_to_ret" in locals() and years_to_ret > 0 and len(burden_list) == years_to_ret:
            df_burden_path = pd.DataFrame(
                {
                    "Jahr": year_list,
                    "Belastungsindex": burden_list,
                    "Rendite-Faktor": factor_list,
                }
            )

            col_b1, col_b2 = st.columns(2)

            col_b1.line_chart(
                df_burden_path.set_index("Jahr")[["Belastungsindex"]],
                height=250,
            )
            col_b1.caption("Pfad des Belastungsindex während der verbleibenden Sparphase.")

            col_b2.line_chart(
                df_burden_path.set_index("Jahr")[["Rendite-Faktor"]],
                height=250,
            )
            col_b2.caption("Anpassungsfaktor für die Rendite pro Jahr (ab heute).")

            st.metric(
                label="Durchschnittlicher Belastungsindex (ab heute)",
                value=f"{df_burden_path['Belastungsindex'].mean():.2f}",
            )
            st.metric(
                label="Effektive Rendite r_eff",
                value=f"{r_effective*100:.2f} %",
                delta=f"Basisr.: {r_base*100:.2f} %",
            )
        else:
            st.info("Bitte im Tab «Modell» zuerst Parameter wählen.")

    # ------------------------------------------------------------------
    # TAB 3: SZENARIO-IMPACT (DELTA-FUNKTION)
    # ------------------------------------------------------------------
    with tab_compare:
        st.subheader("Szenario-Impact: Vergleich aller Szenarien")

        st.markdown(
            """
            Hier werden **alle verfügbaren Szenarien** mit den gleichen Eingaben  
            (Alter, Startjahr, Lohn, Koordinationsabzug, Beitragssatz, gewählter UWS, Elastizität) durchgerechnet.  
            Das aktuell im Tab *Modell* gewählte Szenario dient als **Baseline** für die Deltas.

            Für alle Szenarien gilt:
            - Sparbeginn: Alter 25  
            - Sparende: gewähltes Pensionierungsalter  
            - Gleiches Lohn- und Beitragsprofil
            """
        )

        if "years_to_ret" not in locals() or years_to_ret <= 0:
            st.info("Bitte zuerst im Tab «Modell» die Eingaben festlegen.")
        else:
            # Helferfunktion: ein Szenario komplett durchrechnen
            def compute_for_scenario(s_code: str, s_params: dict):
                # Belastungsindex-Pfad (ab heute bis Pensionierung)
                local_burdens = []
                for i in range(years_to_ret):
                    y = start_year + i
                    b = get_burden_index_from_df(s_code, y, burden_df)
                    local_burdens.append(b)

                if local_burdens:
                    mean_burden = sum(local_burdens) / len(local_burdens)
                else:
                    mean_burden = float("nan")

                # Rendite-Faktoren
                local_factors = [
                    adjust_factor_from_burden(b, ref_burden=0.5, elasticity=elasticity)
                    for b in local_burdens
                ]
                avg_factor_s = (
                    sum(local_factors) / len(local_factors) if local_factors else 1.0
                )

                r_base_s = s_params["return_active"]
                r_eff_s = r_base_s * avg_factor_s

                # Kapital über gesamte Sparphase ab 25
                capital_s = compute_capital_at_retirement(
                    salary=insured_salary,
                    contrib_rate=contrib,
                    years_to_retirement=years_total,
                    return_rate=r_eff_s,
                    contrib_factor=s_params["contrib_factor"],
                )

                uws_fair_s = fair_conversion_rate(
                    life_expectancy_years=le,
                    discount_rate=s_params["discount_retired"],
                )

                pension_chosen_s = capital_s * uws_chosen
                pension_fair_s = capital_s * uws_fair_s

                return {
                    "ScenarioCode": s_code,
                    "Szenario": s_params["name"],
                    "Kapital": capital_s,
                    "FaireRente": pension_fair_s,
                    "GewaehlteRente": pension_chosen_s,
                    "FairerUWS": uws_fair_s,
                    "Belastungsindex_mittel": mean_burden,
                    "Rendite_effektiv": r_eff_s,
                    "Rendite_Basis": r_base_s,
                }

            results = []
            for sc_code, s_params in scenario_params.items():
                try:
                    res = compute_for_scenario(sc_code, s_params)
                    results.append(res)
                except Exception as e:
                    # Falls ein Szenario für die gewählten Jahre nicht funktioniert:
                    st.warning(f"Szenario {sc_code} konnte nicht berechnet werden: {e}")

            if not results:
                st.error("Keine Szenarien konnten berechnet werden.")
            else:
                df_res = pd.DataFrame(results)

                # Baseline = aktuell im Modell-Tab gewähltes Szenario
                baseline_code = scen_code
                if baseline_code in df_res["ScenarioCode"].values:
                    base_row = df_res[df_res["ScenarioCode"] == baseline_code].iloc[0]
                    base_kap = base_row["Kapital"]
                    base_fair = base_row["FaireRente"]
                    base_chosen = base_row["GewaehlteRente"]
                    base_uws = base_row["FairerUWS"]
                    base_burden = base_row["Belastungsindex_mittel"]
                    base_r_eff = base_row["Rendite_effektiv"]

                    df_res["ΔKapital"] = df_res["Kapital"] - base_kap
                    df_res["ΔFaireRente"] = df_res["FaireRente"] - base_fair
                    df_res["ΔGewaehlteRente"] = df_res["GewaehlteRente"] - base_chosen
                    df_res["ΔFairerUWS_pp"] = (df_res["FairerUWS"] - base_uws) * 100
                    df_res["ΔBelastungsindex"] = df_res["Belastungsindex_mittel"] - base_burden
                    df_res["ΔRendite_eff_pp"] = (df_res["Rendite_effektiv"] - base_r_eff) * 100
                else:
                    st.warning(
                        "Das aktuelle Basisszenario wurde in den Resultaten nicht gefunden – "
                        "Deltas werden nicht berechnet."
                    )

                st.markdown("### Übersicht: absolute Werte je Szenario")
                st.dataframe(
                    df_res[
                        [
                            "ScenarioCode",
                            "Szenario",
                            "Kapital",
                            "FaireRente",
                            "GewaehlteRente",
                            "FairerUWS",
                            "Belastungsindex_mittel",
                            "Rendite_effektiv",
                        ]
                    ].sort_values("ScenarioCode"),
                    use_container_width=True,
                )

                if "ΔKapital" in df_res.columns:
                    st.markdown(f"### Deltas relativ zum Basisszenario: **{scen_name}**")
                    st.dataframe(
                        df_res[
                            [
                                "ScenarioCode",
                                "Szenario",
                                "ΔKapital",
                                "ΔFaireRente",
                                "ΔGewaehlteRente",
                                "ΔFairerUWS_pp",
                                "ΔBelastungsindex",
                                "ΔRendite_eff_pp",
                            ]
                        ].sort_values("ScenarioCode"),
                        use_container_width=True,
                    )

                    st.markdown("### Visualisierung: Jahresrenten nach Szenario")
                    chart_scen = df_res.set_index("Szenario")[["GewaehlteRente", "FaireRente"]]
                    st.bar_chart(chart_scen)

    # ------------------------------------------------------------------
    # TAB 4: METHODIK
    # ------------------------------------------------------------------
    with tab_method:
        st.subheader("Methodische Hinweise (Kurzfassung)")

        st.markdown(
            f"""
            1. **Bevölkerungsszenarien (BFS)**  
               * Tabelle px-x-0104000000_102  
               * Aggregation 20–64 (Erwerbstätige) und 65+ (Pensionierte)  
               * Belastungsindex = 65+ / 20–64 je Szenario & Jahr  

            2. **Belastungsindex → Rendite in der Sparphase**  
               * Für jedes Sparjahr ab heutigem Alter wird der Belastungsindex B(t) ermittelt  
               * Jahr für Jahr wird ein Rendite-Faktor  
                 \\( f(t) = 1 − \\text{{Elastizität}} \\cdot (B(t) − B_{{ref}}) \\) berechnet  
               * \\( r_{{eff}} = r_{{Basis}} \\cdot \\overline{{f(t)}} \\)  
               * r_eff wird für die **gesamte Sparphase ab Alter {SAVING_START_AGE}** verwendet
                 (didaktische Vereinfachung).

            3. **Rentenphase / fairer Umwandlungssatz**  
               * Lebenserwartung (ex) ab dem gewählten Pensionierungsalter
                 aus px-x-0102020300_102  
               * UWS_fair = 1 / Rentenbarwertfaktor (nachschüssig)  

            4. **Koordinationsabzug**  
               * BVG-pflichtiger Lohn = max(0, Bruttolohn − Koordinationsabzug)  
               * Sämtliche Beiträge werden auf Basis dieses BVG-Lohnes berechnet.  

            5. **Didaktischer Charakter**  
               * Modell ist bewusst vereinfacht (kein genauer Lohnpfad,
                 keine versicherungstechnischen Reserven),
                 dient aber zur Illustration der Richtungseffekte aus
                 deiner schriftlichen Analyse im CAS Datenkompetenz.
            """
        )


if __name__ == "__main__":
    main()
