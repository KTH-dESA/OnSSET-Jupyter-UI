from IPython.display import display, Markdown, HTML
from ipywidgets import widgets
import pandas as pd
import numpy as np
from collections import defaultdict
from math import ceil, sqrt, pi, exp, log, copysign
from pyproj import Proj
import seaborn as sns
import matplotlib.pylab as plt
import folium
import ipyleaflet as ll
from pyproj import Proj

SET_COUNTRY = 'Country'  # This cannot be changed, lots of code will break
SET_X = 'X'  # Coordinate in kilometres
SET_Y = 'Y'  # Coordinate in kilometres
SET_X_DEG = 'X_deg'
SET_Y_DEG = 'Y_deg'
SET_POP = 'Pop'  # Population in people per point (equally, people per km2)
SET_POP_CALIB = 'Pop2015Act'  # Calibrated population to reference year, same units
SET_POP_FUTURE = 'Pop2030'  # Project future population, same units
SET_GRID_DIST_CURRENT = 'GridDistCurrent'  # Distance in km from current grid
SET_GRID_DIST_PLANNED = 'GridDistPlan'  # Distance in km from current and future grid
SET_ROAD_DIST = 'RoadDist'  # Distance in km from road network
SET_NIGHT_LIGHTS = 'NightLights'  # Intensity of night time lights (from NASA), range 0 - 63
SET_TRAVEL_HOURS = 'TravelHours'  # Travel time to large city in hours
SET_GHI = 'GHI'  # Global horizontal irradiance in kWh/m2/day
SET_WINDVEL = 'WindVel'  # Wind velocity in m/s
SET_WINDCF = 'WindCF'  # Wind capacity factor as percentage (range 0 - 1)
SET_HYDRO = 'Hydropower'  # Hydropower potential in kW
SET_HYDRO_DIST = 'HydropowerDist'  # Distance to hydropower site in km
SET_SUBSTATION_DIST = 'SubstationDist'
SET_ELEVATION = 'Elevation'
SET_SLOPE = 'Slope'
SET_LAND_COVER = 'LandCover'
SET_SOLAR_RESTRICTION = 'SolarRestriction'
SET_ROAD_DIST_CLASSIFIED = 'RoadDistClassified'
SET_SUBSTATION_DIST_CLASSIFIED = 'SubstationDistClassified'
SET_ELEVATION_CLASSIFIED = 'ElevationClassified'
SET_SLOPE_CLASSIFIED = 'SlopeClassified'
SET_LAND_COVER_CLASSIFIED = 'LandCoverClassified'
SET_COMBINED_CLASSIFICATION = 'GridClassification'
SET_GRID_PENALTY = 'GridPenalty'
SET_URBAN = 'IsUrban'  # Whether the site is urban (0 or 1)
SET_ELEC_PREFIX = 'Elec'
SET_ELEC_CURRENT = 'Elec2015'  # If the site is currently electrified (0 or 1)
SET_ELEC_FUTURE = 'Elec2030'  # If the site has the potential to be 'easily' electrified in future
SET_NEW_CONNECTIONS = 'NewConnections'  # Number of new people with electricity connections
SET_MIN_GRID_DIST = 'MinGridDist'
SET_LCOE_GRID = 'lcoe_grid'  # All lcoes in USD/kWh
SET_LCOE_SA_PV = 'lcoe_sa_pv'
SET_LCOE_SA_DIESEL = 'lcoe_sa_diesel'
SET_LCOE_MG_WIND = 'lcoe_mg_wind'
SET_LCOE_MG_DIESEL = 'lcoe_mg_diesel'
SET_LCOE_MG_PV = 'lcoe_mg_pv'
SET_LCOE_MG_HYDRO = 'lcoe_mg_hydro'
SET_MINIMUM_TECH = 'minimum_tech'  # The technology with lowest lcoe (excluding grid)
SET_MINIMUM_OVERALL = 'minimum_overall'
SET_MINIMUM_TECH_LCOE = 'minimum_tech_lcoe'  # The lcoe value
SET_MINIMUM_OVERALL_LCOE = 'minimum_overall_lcoe'
SET_MINIMUM_OVERALL_CODE = 'minimum_overall_code'
SET_MINIMUM_CATEGORY = 'minimum_category'  # The category with minimum lcoe (grid, minigrid or standalone)
SET_NEW_CAPACITY = 'NewCapacity'  # Capacity in kW
SET_INVESTMENT_COST = 'InvestmentCost'  # The investment cost in USD
SUM_POPULATION_PREFIX = 'population_'
SUM_NEW_CONNECTIONS_PREFIX = 'new_connections_'
SUM_CAPACITY_PREFIX = 'capacity_'
SUM_INVESTMENT_PREFIX = 'investment_'

def print_summary(summary):
    display(Markdown('### Summaries  \nHere we see the summaries of the model run.'))
    index = ['Grid', 'SA Diesel', 'SA PV', 'MG Wind', 'MG Diesel', 'MG PV', 'MG Hydro', 'Total']
    columns = ['Population', 'New connections', 'Capacity (kw)', 'Investments (million USD)']
    tab = pd.DataFrame(index=index, columns=columns)

    tab[columns[0]] = summary.iloc[0:7].astype(int).tolist() + [int(summary.iloc[0:7].sum())]
    tab[columns[1]] = summary.iloc[7:14].astype(int).tolist() + [int(summary.iloc[7:14].sum())]
    tab[columns[2]] = summary.iloc[14:21].astype(int).tolist() + [int(summary.iloc[14:21].sum())]
    tab[columns[3]] = [round(x/1e4)/1e2 for x in summary.iloc[21:28].astype(float).tolist()] + [round(summary.iloc[21:28].sum()/1e4)/1e2]
    return tab