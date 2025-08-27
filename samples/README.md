## Sample datasets

This folder contains small, self‑contained dataset files. Most files are sampled subsets of larger datasets and do not represent the full source data.

### AEMO NEM aggregated price and demand (CSV)

Files: `aemo_price_demand/*.csv`  
See also: `aemo_price_demand/README.md`

These CSV files contain aggregated electricity price and demand observations for selected National Electricity Market (NEM) regions. Each row reports a settlement timestamp for a region together with total demand and the regional reference price.

Columns include: `REGION` (region identifier, e.g., NSW1, QLD1), `SETTLEMENTDATE` (timestamp), `TOTALDEMAND` (aggregated demand), `RRP` (regional reference price), and `PERIODTYPE` (e.g., TRADE).

### Fluvius electricity smart meters – 15‑minute values (CSV)

Files: `fluvius_smart_meter/*.csv`  
See also: `fluvius_smart_meter/README.md`

These files contain 15‑minute interval consumption and injection values from anonymized residential electricity meters in Flanders. The samples are split into categories (with/without solar panels, heat pumps, and electric vehicles); each file represents one category.

Columns include: `EAN_ID` (anonymized meter ID), `Datum` (date), `Datum_Startuur` (interval start timestamp), `Volume_Afname_KWh` (consumption in kWh), `Volume_Injectie_KWh` (injection in kWh), `Warmtepomp_Indicator` (0/1), `Elektrisch_Voertuig_Indicator` (0/1), `PV_Installatie_Indicator` (0/1), `Contract_Categorie` (e.g., Residentieel).

### Fluvius gas meters – hourly values (CSV)

Files: `fluvius_gas_meter/*.csv`  
See also: `fluvius_gas_meter/README.md`

Hourly gas consumption samples from anonymized residential meters. Each row covers one hour for one meter, with start and end timestamps and measured energy volume.

Columns include: `EAN_ID`, `Datum` (date), `Datum_Startuur` (hour start), `Datum_Einduur` (hour end), `Volume_Afname_KWh` (consumption in kWh), `Type_Gasmeter` (e.g., G4, G6), `Contract_Categorie` (e.g., Residentieel).

### Belgium electricity smart meter sample (SQLite)

File: `belgium_electricity_smartmeter_sample.sqlite`

SQLite database containing a sample of 15‑minute electricity meter data. The main table `measurement_data` mirrors the electricity CSV structure for convenient querying; a small text table stores accompanying notes.

Tables:
- `measurement_data` with columns: `EAN_ID`, `Datum`, `Datum_Startuur`, `Volume_Afname_KWh`, `Volume_Injectie_KWh`, `Warmtepomp_Indicator`, `Elektrisch_Voertuig_Indicator`, `PV_Installatie_Indicator`, `Contract_Categorie`, `source_file`.
- `legende_txt(id, content)` for descriptive text.

### Belgium gas meter sample (SQLite)

File: `belgium_gasmeter_sample.sqlite`

SQLite database containing an hourly gas consumption sample. The `measurement_data` table mirrors the gas CSV structure for querying; a small text table stores accompanying info.

Tables:
- `measurement_data` with columns: `EAN_ID`, `Datum`, `Datum_Startuur`, `Datum_Einduur`, `Volume_Afname_KWh`, `Type_Gasmeter`, `Contract_Categorie`.
- `info_txt(id, content)` for descriptive text.

### Croatia public buildings energy (SQLite)

File: `croatia_public_buildings.sqlite`

SQLite database with monthly energy use and cost records for public buildings. Each row represents a building/month/energy‑source combination with measured quantities, energy content, prices, and emissions.

Table:
- `Energy_data` with columns: `ISGE object code`, `Type of object`, `Year`, `Month`, `Energy source`, `Quantity`, `kWh`, `Price in euros`, `Price with VAT in euros`, `CO2`, `Primary energy`, `City/Location`, `Useful surface area Ak`, `Unit of measurement`.

### Longtable CAPEX (Excel)

File: `longtable_capex.xlsx`

An Excel workbook with one sheet (`Tabelle1`) listing equipment/system components and associated technical and price attributes. Each row describes a component or configuration with fields for performance and pricing.

Columns include: `Country`, `Source`, `System type`, `Component`, `Price includes (Material, Labour, Taxes, Scaffolding, OPEX)`, `Material`, `Thickness [cm]`, `Lambda [W/mK]`, `Uw [W/m²K]`, `sCOP`, `Power [kW]`, `T emission [°C]`, `Efficiency`, `Collectors`, `Panel size [m²]`, `Capacity [L]`, `Single/double`, `Price [€/m²]`, `Price [€]`, `Comment`.


### SmartMeter energy use in London households (CSV, Excel)

Files: `london_smartmeter_energy_data/CC_LCL-FullData_sample.csv`, `london_smartmeter_energy_data/Tariffs.xlsx`  
See also: `london_smartmeter_energy_data/README.md`

Half‑hourly household electricity consumption readings for London households. The CSV contains anonymized household IDs, tariff group labels, timestamps, and consumption values in kWh per half‑hour. An accompanying Excel workbook lists the dynamic time‑of‑use price signal schedule used in the 2013 trial (High/Low/Normal) with applicable dates/times.

Columns include (CSV): `LCLid` (household ID), `stdorToU` (standard vs time‑of‑use group), `DateTime` (timestamp), `KWH/hh (per half hour)` (consumption in kWh).


### Three years of hourly data from Danish smart heat meters (CSV)

Files: `danish_smart_heat_meter_data/processed_final_data_sample.csv`, `danish_smart_heat_meter_data/contextual_data.csv`  
See also: `danish_smart_heat_meter_data/README.md`

Processed hourly heat consumption series and contextual building metadata. The processed sample provides equidistant hourly measurements per meter, while the contextual file lists meter/building attributes.

Columns include (processed sample): `customer_id`, `meter_id`, `time_rounded` (UTC timestamp), `energy_heat_kwh`, `volume_flow_m3`, `inlet_flow_energy_m3xC`, `back_flow_energy_m3xC`.

Columns include (contextual): `meter_id`, `customer_id`, `unit_type`, `construction_year`, `energy_label`.


### Lower Saxony residential electric load (HDF5)

Files: `lower_saxony_electric_load/2018_data_60min.hdf5`, `lower_saxony_electric_load/2018_data_spatial.hdf5`, `lower_saxony_electric_load/2018_district_heating_grid.hdf5`, `lower_saxony_electric_load/2018_weather.hdf5`  
See also: `lower_saxony_electric_load/README.md`

Electric load measurements from 38 households in Lower Saxony, Germany, provided at multiple temporal resolutions and spatial aggregations. These HDF5 files contain 2018 subsets, including 60‑minute time‑series, spatial aggregations, district heating grid information, and weather variables used alongside the load data.

