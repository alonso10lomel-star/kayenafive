# kayena
Adapting Farms
# Crop Rotation with NASA POWER Data

Kayena is building a prototype that takes a location and uses NASA POWER climate data plus agronomic sources (FAO and INIFAP) to compare three crop rotation alternatives and explain why each one does or does not make sense. The test case is in the state of Chihuahua, Mexico (latitude 29.25, longitude -107.41), with daily data from 2016 to 2025. This is a NASA Space Apps project (November 14–15, 2026). Our success criterion is an honest prototype: it compares options using rules that can be traced back to their source, and it clearly states what it does not know.

## Problem and User

A farmer deciding what to plant after the current crop rarely has the area's climate history at hand, or a clear way to compare options. Planting the same crop again and again wears out the soil, and choosing a crop that cannot handle the region's temperature or water is costly.

- **User:** a farmer or agricultural technician in a semi-arid region of northern Mexico.
- **What they give the system:** their location, the previous crop, and their priority (for example, saving water).
- **What they get:** three possible rotations, each with its reason, the condition that favors it, and its limitation.

## Objectives

**General objective:** compare three crop rotation alternatives for a location, using NASA POWER climate data and sourced agronomic criteria, and explain each result along with its limitations.

**Specific MVP objectives :**

1. Obtain and clean a daily NASA POWER CSV with temperature, precipitation, and humidity, documenting source, time period, and missing values.
2. Calculate basic climate indicators, such as average temperature and accumulated rainfall.
3. Define three candidate crops and three criteria traceable to FAO or INIFAP: temperature, water, and season.
4. Show in an app the flow input → data → three rotations, with at least one reason per option.
5. Store the data in a private bucket.

**Extensions :**

1. Connect a local reading from an ESP32 sensor, labeled as a practice demo.
2. Compare a small TinyML model (GMM) against the rules, without presenting it as a validated prediction.
3. Publish the app online and have a backup video.

## Our Data

The main source is a daily CSV from [NASA POWER](https://power.larc.nasa.gov/) for latitude 29.25 and longitude -107.41 (Chihuahua), from January 1, 2016 to December 31, 2025. That is about 3,650 days, one per row.

| Column | What it measures | Unit | What we use it for |
| --- | --- | --- | --- |
| `YEAR`, `DOY` | Year and day of year (1 to 365) | — | Build the date and split seasons |
| `T2M` | Mean temperature at 2 m | °C | Whether the crop tolerates the local climate |
| `T2M_MAX`, `T2M_MIN` | Maximum and minimum temperature at 2 m | °C | Risk of extreme heat and frost |
| `T2MDEW` | Dew/frost point at 2 m | °C | Support for estimating humidity and frost |
| `RH2M` | Relative humidity at 2 m | % | Humidity context and pest risk |
| `PRECTOTCORR` | Corrected precipitation | mm/day | Accumulated rainfall per season, the water criterion |

- **Format:** the file starts with a NASA header that ends at `-END HEADER-`. Those lines must be skipped when reading it with pandas.
- **Missing values:** NASA marks days without data as `-999`. They are replaced with empty values before computing averages.
- **Spatial scope:** this is regional gridded data, not a field measurement. It describes the area, not the soil of a specific farm.
- **Agronomic sources:** each crop's temperature, water, and season requirements come from [FAO](https://www.fao.org/) and [INIFAP](https://www.gob.mx/inifap). Anything without a source is marked as illustrative.
- **Local sensor (extension):** ESP32 readings with variable name and unit. They are presented as a demo, not as calibrated soil moisture.

## How It Works

The user provides three inputs and receives three explained rotations. Everything the app shows can be traced back to the NASA CSV or to an agronomic source.

1. **Input:** the user chooses the location, the previous crop, and their priority.
2. **Data:** the app loads the cleaned NASA POWER CSV for that area and calculates indicators such as average temperature, frost risk, and accumulated rainfall per season.
3. **Criteria:** it compares each candidate crop against three sourced rules (temperature, water, and season).
4. **Output:** it shows three rotations, each with its reason, the condition that favors it, and its limitation.
5. **Extensions:** if time allows, we add a local ESP32 reading and an experimental TinyML comparison.



## Team and Roles

| Member | Role | Contribution to the project |
| --- | --- | --- |
| Kevin Alejandro Chávez Juárez | Coordination and technical integration | Scope, plan B decisions, and final dry run |
| Daniel Sebastian Torres Ontiveros | NASA data analysis | Clean NASA POWER CSV, indicators, and variable dictionary |
| César Yair Chavez Frias | Agronomic research and validation | Candidate crops, three rotations, and criteria sourced from FAO or INIFAP |
| Sebastian Alonso Lopez Melendez | AWS/Jupyter, shared data, and communication | S3 bucket, Jupyter data reading, evidence organization, user guide, and pitch |
| Oscar Ramses Carrasco Cordova | App and TinyML/GMM testing | Application, rule-based comparison, and experimental GMM test |
| Misael Arturo Zepeda Juarez | ESP32, sensors, and physical testing | Local sensor reading and physical demo |

