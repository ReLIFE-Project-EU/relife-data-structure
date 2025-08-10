### Sample datasets overview

### belgium_electricity_smartmeter_sample.sqlite

- **format**: SQLite database
- **tables**:
  - **measurement_data** (1,004,803 rows)
    - **EAN_ID**: anonymized meter identifier
    - **Datum**: date
    - **Datum_Startuur**: start timestamp (ISO 8601)
    - **Volume_Afname_KWh**: electricity consumption for the following 15-minute interval, in kWh
    - **Volume_Injectie_KWh**: electricity injection for the following 15-minute interval, in kWh
    - **Warmtepomp_Indicator**: heat pump indicator (0/1)
    - **Elektrisch_Voertuig_Indicator**: electric vehicle indicator (0/1)
    - **PV_Installatie_Indicator**: photovoltaic installation indicator (0/1)
    - **Contract_Categorie**: contract category (e.g., "Residentieel")
    - **source_file**: source CSV filename per record
  - **legende_txt** (1 row)
    - **content**: provider’s textual description and field notes
- **notes**:
  - Indicators are stored as 0/1 values.
  - Timestamps in samples appear in ISO 8601 with `Z` suffix.

### belgium_gasmeter_sample.sqlite

- **format**: SQLite database
- **tables**:
  - **measurement_data** (895,185 rows)
    - **EAN_ID**: anonymized meter identifier
    - **Datum**: date
    - **Datum_Startuur**: start timestamp (ISO 8601)
    - **Datum_Einduur**: end timestamp (ISO 8601)
    - **Volume_Afname_KWh**: gas consumption for the interval, in kWh
    - **Type_Gasmeter**: gas meter type (observed values: `G4`, `G6`)
    - **Contract_Categorie**: contract category (e.g., "Residentieel")
  - **info_txt** (1 row)
    - **content**: provider’s textual description and field notes
- **notes**:
  - Hourly intervals indicated by start and end timestamps.

### croatia_public_buildings.sqlite

- **format**: SQLite database
- **tables**:
  - **Energy_data** (1,209,674 rows)
    - **ISGE object code**: facility/object code
    - **Type of object**: object type (e.g., "University building")
    - **Year**: year
    - **Month**: month (1–12)
    - **Energy source**: source/fuel (e.g., Electric energy, Water, Extra light fuel oil)
    - **Quantity**: reported quantity (unit varies by source)
    - **kWh**: energy in kWh
    - **Price in euros**: price excluding VAT, in EUR
    - **Price with VAT in euros**: price including VAT, in EUR
    - **CO2**: emissions value
    - **Primary energy**: primary energy value
    - **City/Location**: location name (e.g., Opatija)
    - **Useful surface area Ak**: reported surface area
    - **Unit of measurement**: unit string (e.g., kWh, m³, l)
- **notes**:
  - Units vary by energy source; both quantity and normalized kWh are present.

### longtable_capex.xlsx

- **format**: Excel workbook
- **sheets**: `Tabelle1`, `microsoft.com:RD`, `microsoft.com:Single`, `microsoft.com:FV`, `microsoft.com:CNMTM`, `microsoft.com:LET_WF`, `microsoft.com:LAMBDA_WF`, `microsoft.com:ARRAYTEXT_WF`
- **observations**:
  - `Tabelle1` contains numeric values in the first few rows (raw XML preview showed sequences like 0–27 and some decimal/scientific values). Column headers were not detected in the quick preview.
  - The `microsoft.com:*` sheets appear to be function/helper sheets present in some workbooks; no tabular preview extracted here.


