# Sample Datasets

This directory contains small, realistic datasets for energy and building analytics. Below is an overview of each file.

### `belgium_electricity_smartmeter_sample.sqlite`
- Purpose: Hourly electricity smart‑meter readings for Belgian customers.
- Contents: `measurement_data` with one row per EAN and hour; `legende_txt` with explanatory text.
- Notable fields: `EAN_ID` (meter identifier), `Datum` (date), `Datum_Startuur` (hour start), `Volume_Afname_KWh` (consumption), `Volume_Injectie_KWh` (injection back to grid), binary indicators for `Warmtepomp_Indicator`, `Elektrisch_Voertuig_Indicator`, `PV_Installatie_Indicator`, `Contract_Categorie`, and `source_file`.
- Notes: Indicators are encoded as 0/1. Timestamps are text as provided by the source; no timezone normalization is applied in this sample.

### `belgium_gasmeter_sample.sqlite`
- Purpose: Hourly gas meter readings for Belgian customers.
- Contents: `measurement_data` with one row per EAN and hour; `info_txt` with explanatory text.
- Notable fields: `EAN_ID`, `Datum`, `Datum_Startuur`, `Datum_Einduur`, `Volume_Afname_KWh` (gas energy use), `Type_Gasmeter` (meter type), `Contract_Categorie`.
- Notes: Time fields are stored as text. No weather normalization or temperature data are included.

### `croatia_public_buildings.sqlite`
- Purpose: Monthly energy consumption and cost records for Croatian public buildings.
- Contents: Single table `Energy_data` covering multiple energy sources and cost metrics across years and months.
- Notable fields: `Year`, `Month`, `Energy source`, `Quantity`, `kWh`, `Price in euros`, `Price with VAT in euros`, `CO2`, `Primary energy`, `City/Location`, `Useful surface area Ak`, `Unit of measurement` (e.g., `kWh`, `m³`).
- Notes: Large dataset (~1.21M rows). Useful for aggregation, benchmarking, and cost/emissions analysis by location and building type.

### `longtable_capex.xlsx`
- Purpose: Reference CAPEX items for building systems and components.
- Contents: Sheet `Tabelle1` with 541 rows of line‑items (materials, labor, and equipment).
- Notable fields: `Country`, `Source`, `System type`, `Component`, `Material`, `Thickness [cm]`, `Lambda [W/mK]`, `Uw [W/m²K]`, `sCOP`, `Power [kW]`, `Efficiency`, `Price [€/m²]`, `Price [€]`, plus a detailed "Price includes (...)" qualifier.
- Notes: Some rows represent labor or ancillary items; numeric fields may be blank where not applicable. Values are illustrative and should be validated before use in cost models.
