from IPython.display import display, Markdown, HTML
import pandas as pd
import numpy as np
from collections import defaultdict
from math import ceil, sqrt, pi, exp, log, copysign
from pyproj import Proj
import seaborn as sns
import matplotlib.pylab as plt
import folium
import branca.colormap as cm

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
SET_LCOE_GRID = 'grid'  # All lcoes in USD/kWh
SET_LCOE_SA_PV = 'sa_pv'
SET_LCOE_SA_DIESEL = 'sa_diesel'
SET_LCOE_MG_WIND = 'mg_wind'
SET_LCOE_MG_DIESEL = 'mg_diesel'
SET_LCOE_MG_PV = 'mg_pv'
SET_LCOE_MG_HYDRO = 'mg_hydro'
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

LHV_DIESEL = 9.9445485
HOURS_PER_YEAR = 8760
max_grid_extension_dist = 50


def condition(df):
    df.fillna(0, inplace=True)
    df.sort_values(by=[SET_COUNTRY, SET_Y, SET_X], inplace=True)
    project = Proj('+proj=merc +lon_0=0 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs')

    def get_x(row):
        x, y = project(row[SET_X] * 1000, row[SET_Y] * 1000, inverse=True)
        return x

    def get_y(row):
        x, y = project(row[SET_X] * 1000, row[SET_Y] * 1000, inverse=True)
        return y

    df[SET_X_DEG] = df.apply(get_x, axis=1)
    df[SET_Y_DEG] = df.apply(get_y, axis=1)

    return df


def grid_penalties(df):
    def classify_road_dist(row):
        road_dist = row[SET_ROAD_DIST]
        if road_dist <= 5:
            return 5
        elif road_dist <= 10:
            return 4
        elif road_dist <= 25:
            return 3
        elif road_dist <= 50:
            return 2
        else:
            return 1

    def classify_substation_dist(row):
        substation_dist = row[SET_SUBSTATION_DIST]
        if substation_dist <= 1:
            return 5
        elif substation_dist <= 5:
            return 4
        elif substation_dist <= 10:
            return 3
        elif substation_dist <= 50:
            return 2
        else:
            return 1

    def classify_land_cover(row):
        land_cover = row[SET_LAND_COVER]
        if land_cover == 0:
            return 1
        elif land_cover == 1:
            return 3
        elif land_cover == 2:
            return 2
        elif land_cover == 3:
            return 3
        elif land_cover == 4:
            return 2
        elif land_cover == 5:
            return 3
        elif land_cover == 6:
            return 4
        elif land_cover == 7:
            return 5
        elif land_cover == 8:
            return 4
        elif land_cover == 9:
            return 5
        elif land_cover == 10:
            return 5
        elif land_cover == 11:
            return 1
        elif land_cover == 12:
            return 3
        elif land_cover == 13:
            return 3
        elif land_cover == 14:
            return 5
        elif land_cover == 15:
            return 3
        elif land_cover == 16:
            return 5

    def classify_elevation(row):
        elevation = row[SET_SUBSTATION_DIST]
        if elevation <= 500:
            return 5
        elif elevation <= 1000:
            return 4
        elif elevation <= 2000:
            return 3
        elif elevation <= 3000:
            return 2
        else:
            return 1

    def classify_slope(row):
        slope = row[SET_SUBSTATION_DIST]
        if slope <= 10:
            return 5
        elif slope <= 20:
            return 4
        elif slope <= 30:
            return 3
        elif slope <= 40:
            return 2
        else:
            return 1

    def set_penalty(row):
        classification = row[SET_COMBINED_CLASSIFICATION]
        if classification <= 2:
            return 1.00
        elif classification <= 3:
            return 1.02
        elif classification <= 4:
            return 1.05
        else:
            return 1.10

    df[SET_ROAD_DIST_CLASSIFIED] = df.apply(classify_road_dist, axis=1)
    df[SET_SUBSTATION_DIST_CLASSIFIED] = df.apply(classify_substation_dist, axis=1)
    df[SET_LAND_COVER_CLASSIFIED] = df.apply(classify_land_cover, axis=1)
    df[SET_ELEVATION_CLASSIFIED] = df.apply(classify_elevation, axis=1)
    df[SET_SLOPE_CLASSIFIED] = df.apply(classify_slope, axis=1)

    df[SET_COMBINED_CLASSIFICATION] = (0.05 * df[SET_ROAD_DIST_CLASSIFIED] +
                                       0.09 * df[SET_SUBSTATION_DIST_CLASSIFIED] +
                                       0.39 * df[SET_LAND_COVER_CLASSIFIED] +
                                       0.15 * df[SET_ELEVATION_CLASSIFIED] +
                                       0.32 * df[SET_SLOPE_CLASSIFIED])

    df[SET_GRID_PENALTY] = df.apply(set_penalty, axis=1)

    return df


def wind(df):
    mu = 0.97  # availability factor
    t = 8760
    p_rated = 600
    z = 55  # hub height
    zr = 80  # velocity measurement height
    es = 0.85  # losses in wind electricty
    u_arr = range(1, 26)
    p_curve = [0, 0, 0, 0, 30, 77, 135, 208, 287, 371, 450, 514, 558,
               582, 594, 598, 600, 600, 600, 600, 600, 600, 600, 600, 600]

    def get_wind_cf(row):
        u_zr = row[SET_WINDVEL]
        if u_zr == 0:
            return 0

        else:
            # Adjust for the correct hub height
            alpha = (0.37 - 0.088 * log(u_zr)) / (1 - 0.088 * log(zr / 10))
            u_z = u_zr * (z / zr) ** alpha

            # Rayleigh distribution and sum of series
            rayleigh = [(pi / 2) * (u / u_z ** 2) * exp((-pi / 4) * (u / u_z) ** 2) for u in u_arr]
            energy_produced = sum([mu * es * t * p * r for p, r in zip(p_curve, rayleigh)])

            return energy_produced / (p_rated * t)

    df[SET_WINDCF] = df.apply(get_wind_cf, axis=1)

    return df


def pop(df, pop_2015, urban_ratio_2015, pop_2030, urban_ratio_2030):
    # Calculate the ratio between the actual population and the total population from the GIS layer
    pop_actual = pop_2015
    pop_sum = df[SET_POP].sum()
    pop_ratio = pop_actual / pop_sum

    # And use this ratio to calibrate the population in a new column
    df[SET_POP_CALIB] = df.apply(lambda row: row[SET_POP] * pop_ratio, axis=1)

    # Calculate the urban split, by calibrating the cutoff until the target ratio is achieved
    # Keep looping until it is satisfied or another break conditions is reached
    target = urban_ratio_2015
    cutoff = 500  # Start with a cutoff value from specs
    calculated = 0
    count = 0
    prev_vals = []  # Stores cutoff values that have already been tried to prevent getting stuck in a loop
    accuracy = 0.05
    max_iterations = 30
    while True:
        # Assign the 1 (urban)/0 (rural) values to each cell
        df[SET_URBAN] = df.apply(lambda row: 1 if row[SET_POP_CALIB] > cutoff else 0, axis=1)

        # Get the calculated urban ratio, and limit it to within reasonable boundaries
        pop_urb = df.loc[df[SET_URBAN] == 1, SET_POP_CALIB].sum()
        calculated = pop_urb / pop_actual

        if calculated == 0:
            calculated = 0.05
        elif calculated == 1:
            calculated = 0.999

        if abs(calculated - target) < accuracy:
            break
        else:
            cutoff = sorted([0.5, cutoff - cutoff * (target - calculated) / target, 1000000.0])[1]

        if cutoff in prev_vals:
            print('pop NOT SATISFIED: repeating myself')
            break
        else:
            prev_vals.append(cutoff)

        if count >= max_iterations:
            print('pop NOT SATISFIED: got to {}'.format(max_iterations))
            break

        count += 1

    # Save the calibrated cutoff and split so they can be compared
    urban_ratio_modelled = calculated

    # Project future population, with separate growth rates for urban and rural
    urban_growth = (urban_ratio_2030 * pop_2030) / (
        urban_ratio_2015 * pop_2015)
    rural_growth = ((1 - urban_ratio_2030) * pop_2030) / (
        (1 - urban_ratio_2015) * pop_2015)

    df[SET_POP_FUTURE] = df.apply(lambda row: row[SET_POP_CALIB] * urban_growth
    if row[SET_URBAN] == 1
    else row[SET_POP_CALIB] * rural_growth,
                                  axis=1)
    return df


def get_grid_lcoe_table(scenario, max_dist, num_people_per_hh, transmission_losses, base_to_peak_load_ratio,
                        grid_price, grid_capacity_investment, project_life):
    people_arr_direct = list(range(1000)) + list(range(1000, 10000, 10)) + list(range(10000, 1000000, 1000))
    elec_dists = range(0, int(max_dist) + 1)
    grid_lcoes = pd.DataFrame(index=elec_dists, columns=people_arr_direct)

    for people in people_arr_direct:
        for additional_mv_line_length in elec_dists:
            grid_lcoes[people][additional_mv_line_length] = get_grid_lcoe(people, scenario, num_people_per_hh, False,
                                                                          transmission_losses,
                                                                          base_to_peak_load_ratio, grid_price,
                                                                          grid_capacity_investment,
                                                                          additional_mv_line_length, project_life)

    return grid_lcoes


def get_grid_lcoe(people, scenario, num_people_per_hh, calc_cap_only, transmission_losses,
                  base_to_peak_load_ratio, grid_price, grid_capacity_investment, additional_mv_line_length,
                  project_life):
    return calc_lcoe(people=people,
                     scenario=scenario,
                     num_people_per_hh=num_people_per_hh,
                     om_of_td_lines=0.03,
                     distribution_losses=transmission_losses,
                     connection_cost_per_hh=125,
                     base_to_peak_load_ratio=base_to_peak_load_ratio,
                     system_life=30,
                     additional_mv_line_length=additional_mv_line_length,
                     grid_price=grid_price,
                     grid=True,
                     grid_capacity_investment=grid_capacity_investment,
                     calc_cap_only=calc_cap_only,
                     project_life=project_life)


def get_mg_hydro_lcoe(people, scenario, num_people_per_hh, calc_cap_only, mv_line_length, capital_cost=5000,
                      project_life=15):
    return calc_lcoe(people=people,
                     scenario=scenario,
                     num_people_per_hh=num_people_per_hh,
                     om_of_td_lines=0.03,
                     capacity_factor=0.5,
                     distribution_losses=0.05,
                     connection_cost_per_hh=100,
                     capital_cost=capital_cost,
                     om_costs=0.02,
                     base_to_peak_load_ratio=1,
                     system_life=30,
                     mv_line_length=mv_line_length,
                     calc_cap_only=calc_cap_only,
                     project_life=project_life)


def get_mg_pv_lcoe(people, scenario, num_people_per_hh, calc_cap_only, ghi, capital_cost=4300, project_life=15):
    return calc_lcoe(people=people,
                     scenario=scenario,
                     num_people_per_hh=num_people_per_hh,
                     om_of_td_lines=0.03,
                     capacity_factor=ghi / HOURS_PER_YEAR,
                     distribution_losses=0.05,
                     connection_cost_per_hh=100,
                     capital_cost=capital_cost,
                     om_costs=0.015,
                     base_to_peak_load_ratio=0.9,
                     system_life=20,
                     calc_cap_only=calc_cap_only,
                     project_life=project_life)


def get_mg_wind_lcoe(people, scenario, num_people_per_hh, calc_cap_only, wind_cf, capital_cost=3000, project_life=15):
    return calc_lcoe(
        people=people,
        scenario=scenario,
        num_people_per_hh=num_people_per_hh,
        om_of_td_lines=0.03,
        capacity_factor=wind_cf,
        distribution_losses=0.05,
        connection_cost_per_hh=100,
        capital_cost=capital_cost,
        om_costs=0.02,
        base_to_peak_load_ratio=0.75,
        system_life=20,
        calc_cap_only=calc_cap_only,
        project_life=project_life)


def get_mg_diesel_lcoe(people, scenario, num_people_per_hh, calc_cap_only, diesel_price, capital_cost=721,
                       project_life=15):
    return calc_lcoe(people=people,
                     scenario=scenario,
                     num_people_per_hh=num_people_per_hh,
                     om_of_td_lines=0.03,
                     capacity_factor=0.7,
                     distribution_losses=0.05,
                     connection_cost_per_hh=100,
                     capital_cost=capital_cost,
                     om_costs=0.1,
                     base_to_peak_load_ratio=0.5,
                     system_life=15,
                     efficiency=0.33,
                     diesel_price=diesel_price,
                     diesel=True,
                     calc_cap_only=calc_cap_only,
                     project_life=project_life)


def get_sa_diesel_lcoe(people, scenario, num_people_per_hh, calc_cap_only, diesel_price, capital_cost=938,
                       project_life=15):
    return calc_lcoe(people=people,
                     scenario=scenario,
                     num_people_per_hh=num_people_per_hh,
                     om_of_td_lines=0,
                     capacity_factor=0.7,
                     distribution_losses=0,
                     connection_cost_per_hh=0,
                     capital_cost=capital_cost,
                     om_costs=0.1,
                     base_to_peak_load_ratio=0.5,
                     system_life=10,
                     efficiency=0.28,
                     diesel_price=diesel_price,
                     diesel=True,
                     standalone=True,
                     calc_cap_only=calc_cap_only,
                     project_life=project_life)


def get_sa_pv_lcoe(people, scenario, num_people_per_hh, calc_cap_only, ghi, capital_cost=5500, project_life=15):
    return calc_lcoe(
        people=people,
        scenario=scenario,
        num_people_per_hh=num_people_per_hh,
        om_of_td_lines=0,
        capacity_factor=ghi / HOURS_PER_YEAR,
        distribution_losses=0,
        connection_cost_per_hh=0,
        capital_cost=capital_cost,
        om_costs=0.012,
        base_to_peak_load_ratio=0.9,
        system_life=15,
        standalone=True,
        calc_cap_only=calc_cap_only,
        project_life=project_life)


def calc_lcoe(people, scenario, num_people_per_hh, om_of_td_lines, distribution_losses, connection_cost_per_hh,
              base_to_peak_load_ratio, system_life, mv_line_length=0, om_costs=0.0, capital_cost=0, capacity_factor=1.0,
              efficiency=1.0, diesel_price=0.0, additional_mv_line_length=0, grid_price=0.0, grid=False, diesel=False,
              standalone=False, grid_capacity_investment=0.0, calc_cap_only=False, project_life=15):
    # To prevent any div/0 error
    if people == 0:
        people = 0.00001

    grid_cell_area = 100  # This was 100, changed to 1 which creates different results but let's go with it
    # people *= grid_cell_area  # To adjust for incorrect grid size above

    mv_line_cost = 9000  # USD/km
    lv_line_cost = 5000  # USD/km
    discount_rate = 0.08  # percent
    mv_line_capacity = 50  # kW/line
    lv_line_capacity = 10  # kW/line
    lv_line_max_length = 30  # km
    hv_line_cost = 53000  # USD/km
    mv_line_max_length = 50  # km
    hv_lv_transformer_cost = 5000  # USD/unit
    mv_increase_rate = 0.1  # percentage

    consumption = people / num_people_per_hh * scenario  # kWh/year
    average_load = consumption * (1 + distribution_losses) / HOURS_PER_YEAR
    peak_load = average_load / base_to_peak_load_ratio

    no_mv_lines = ceil(peak_load / mv_line_capacity)
    no_lv_lines = ceil(peak_load / lv_line_capacity)
    lv_networks_lim_capacity = no_lv_lines / no_mv_lines
    lv_networks_lim_length = ((grid_cell_area / no_mv_lines) / (lv_line_max_length / sqrt(2))) ** 2
    actual_lv_lines = ceil(min([people / num_people_per_hh, max([lv_networks_lim_capacity, lv_networks_lim_length])]))
    hh_per_lv_network = (people / num_people_per_hh) / (actual_lv_lines * no_mv_lines)
    lv_unit_length = sqrt(grid_cell_area / (people / num_people_per_hh)) * sqrt(2) / 2
    lv_lines_length_per_lv_network = 1.333 * hh_per_lv_network * lv_unit_length
    total_lv_lines_length = no_mv_lines * actual_lv_lines * lv_lines_length_per_lv_network
    line_reach = (grid_cell_area / no_mv_lines) / (2 * sqrt(grid_cell_area / no_lv_lines))
    total_length_of_lines = min([line_reach, mv_line_max_length]) * no_mv_lines
    additional_hv_lines = max(
        [0, round(sqrt(grid_cell_area) / (2 * min([line_reach, mv_line_max_length])) / 10, 3) - 1])
    hv_lines_total_length = (sqrt(grid_cell_area) / 2) * additional_hv_lines * sqrt(grid_cell_area)
    num_transformers = ceil(additional_hv_lines + no_mv_lines + (no_mv_lines * actual_lv_lines))
    generation_per_year = average_load * HOURS_PER_YEAR

    # The investment and O&M costs are different for grid and non-grid solutions
    if grid:
        td_investment_cost = hv_lines_total_length * hv_line_cost + \
                             total_length_of_lines * mv_line_cost + \
                             total_lv_lines_length * lv_line_cost + \
                             num_transformers * hv_lv_transformer_cost + \
                             (people / num_people_per_hh) * connection_cost_per_hh + \
                             additional_mv_line_length * (
                                 mv_line_cost * (1 + mv_increase_rate) ** ((additional_mv_line_length / 5) - 1))
        td_om_cost = td_investment_cost * om_of_td_lines
        total_investment_cost = td_investment_cost
        total_om_cost = td_om_cost

    else:
        total_lv_lines_length *= 0 if standalone else 0.75
        mv_total_line_cost = mv_line_cost * mv_line_length
        lv_total_line_cost = lv_line_cost * total_lv_lines_length
        installed_capacity = peak_load / capacity_factor
        capital_investment = installed_capacity * capital_cost
        td_investment_cost = mv_total_line_cost + lv_total_line_cost + (
                                                                           people / num_people_per_hh) * connection_cost_per_hh
        td_om_cost = td_investment_cost * om_of_td_lines
        total_investment_cost = td_investment_cost + capital_investment
        total_om_cost = td_om_cost + (capital_cost * om_costs * installed_capacity)

    # The renewable solutions have no fuel cost
    if diesel:
        fuel_cost = diesel_price / LHV_DIESEL / efficiency
    elif grid:
        fuel_cost = grid_price
    else:
        fuel_cost = 0

    # Perform the time value LCOE calculation
    reinvest_year = 0
    if system_life < project_life:
        reinvest_year = system_life

    year = np.arange(project_life)
    el_gen = generation_per_year * np.ones(project_life)
    el_gen[0] = 0
    discount_factor = (1 + discount_rate) ** year
    investments = np.zeros(project_life)
    investments[0] = total_investment_cost
    if reinvest_year:
        investments[reinvest_year] = total_investment_cost

    salvage = np.zeros(project_life)
    used_life = project_life
    if reinvest_year:
        used_life = project_life - system_life  # so salvage will come from the remaining life after the re-investment
    salvage[-1] = total_investment_cost * (1 - used_life / system_life)

    operation_and_maintenance = total_om_cost * np.ones(project_life)
    operation_and_maintenance[0] = 0
    fuel = el_gen * fuel_cost
    fuel[0] = 0

    # So we also return the total investment cost for this number of people
    if calc_cap_only:
        discounted_investments = investments / discount_factor
        return np.sum(discounted_investments) + grid_capacity_investment * peak_load
    else:
        discounted_costs = (investments + operation_and_maintenance + fuel - salvage) / discount_factor
        discounted_generation = el_gen / discount_factor
        return np.sum(discounted_costs) / np.sum(discounted_generation)


def separate_elec_status(elec_status):
    electrified = []
    unelectrified = []

    for i, status in enumerate(elec_status):
        if status:
            electrified.append(i)
        else:
            unelectrified.append(i)
    return electrified, unelectrified


def get_2d_hash_table(x, y, unelectrified, distance_limit):
    hash_table = defaultdict(lambda: defaultdict(list))
    for unelec_row in unelectrified:
        hash_x = int(x[unelec_row] / distance_limit)
        hash_y = int(y[unelec_row] / distance_limit)
        hash_table[hash_x][hash_y].append(unelec_row)
    return hash_table


def get_unelectrified_rows(hash_table, elec_row, x, y, distance_limit):
    unelec_list = []
    hash_x = int(x[elec_row] / distance_limit)
    hash_y = int(y[elec_row] / distance_limit)

    unelec_list.extend(hash_table.get(hash_x, {}).get(hash_y, []))
    unelec_list.extend(hash_table.get(hash_x, {}).get(hash_y - 1, []))
    unelec_list.extend(hash_table.get(hash_x, {}).get(hash_y + 1, []))

    unelec_list.extend(hash_table.get(hash_x + 1, {}).get(hash_y, []))
    unelec_list.extend(hash_table.get(hash_x + 1, {}).get(hash_y - 1, []))
    unelec_list.extend(hash_table.get(hash_x + 1, {}).get(hash_y + 1, []))

    unelec_list.extend(hash_table.get(hash_x - 1, {}).get(hash_y, []))
    unelec_list.extend(hash_table.get(hash_x - 1, {}).get(hash_y - 1, []))
    unelec_list.extend(hash_table.get(hash_x - 1, {}).get(hash_y + 1, []))
    return unelec_list


def pre_elec(df_country, grid):
    pop = df_country[SET_POP_FUTURE].tolist()
    grid_penalty_ratio = df_country[SET_GRID_PENALTY].tolist()
    status = df_country[SET_ELEC_CURRENT].tolist()
    min_tech_lcoes = df_country[SET_MINIMUM_TECH_LCOE].tolist()
    dist_planned = df_country[SET_GRID_DIST_PLANNED].tolist()

    electrified, unelectrified = separate_elec_status(status)

    for unelec in unelectrified:
        pop_index = pop[unelec]
        if pop_index < 1000:
            pop_index = int(pop_index)
        elif pop_index < 10000:
            pop_index = 10 * round(pop_index / 10)
        else:
            pop_index = 1000 * round(pop_index / 1000)

        grid_lcoe = grid[pop_index][int(grid_penalty_ratio[unelec] * dist_planned[unelec])]
        if grid_lcoe < min_tech_lcoes[unelec]:
            status[unelec] = 1
    return status


def elec_direct(df_country, grid, existing_grid_cost_ratio, max_dist):
    x = df_country[SET_X].tolist()
    y = df_country[SET_Y].tolist()
    pop = df_country[SET_POP_FUTURE].tolist()
    grid_penalty_ratio = df_country[SET_GRID_PENALTY].tolist()
    status = df_country[SET_ELEC_FUTURE].tolist()
    min_tech_lcoes = df_country[SET_MINIMUM_TECH_LCOE].tolist()
    new_lcoes = df_country[SET_LCOE_GRID].tolist()

    cell_path = np.zeros(len(status)).tolist()
    electrified, unelectrified = separate_elec_status(status)

    loops = 1
    while len(electrified) > 0:
        print('Electrification loop {} with {} electrified'.format(loops, len(electrified)))
        loops += 1
        hash_table = get_2d_hash_table(x, y, unelectrified, max_dist)

        changes, new_lcoes, cell_path = compare_lcoes(electrified, new_lcoes, min_tech_lcoes,
                                                      cell_path, hash_table, grid, x, y, pop, grid_penalty_ratio,
                                                      max_dist, existing_grid_cost_ratio)

        electrified = changes[:]
        unelectrified = [x for x in unelectrified if x not in electrified]

    return new_lcoes, cell_path


def compare_lcoes(electrified, new_lcoes, min_tech_lcoes, cell_path, hash_table, grid,
                  x, y, pop, grid_penalty_ratio, max_dist, existing_grid_cost_ratio):
    changes = []
    for elec in electrified:
        unelectrified_hashed = get_unelectrified_rows(hash_table, elec, x, y, max_dist)
        for unelec in unelectrified_hashed:
            prev_dist = cell_path[elec]
            dist = grid_penalty_ratio[unelec] * sqrt((x[elec] - x[unelec]) ** 2 + (y[elec] - y[unelec]) ** 2)
            if prev_dist + dist < max_dist:

                pop_index = pop[unelec]
                if pop_index < 1000:
                    pop_index = int(pop_index)
                elif pop_index < 10000:
                    pop_index = 10 * round(pop_index / 10)
                else:
                    pop_index = 1000 * round(pop_index / 1000)

                grid_lcoe = grid[pop_index][int(dist + existing_grid_cost_ratio * prev_dist)]
                if grid_lcoe < min_tech_lcoes[unelec]:
                    if grid_lcoe < new_lcoes[unelec]:
                        new_lcoes[unelec] = grid_lcoe
                        cell_path[unelec] = dist + prev_dist
                        if unelec not in changes:
                            changes.append(unelec)
    return changes, new_lcoes, cell_path


def run_elec(df, grid_lcoes, grid_price, existing_grid_cost_ratio, max_dist):
    # Calculate 2030 pre-electrification
    df[SET_ELEC_FUTURE] = df.apply(lambda row: 1 if row[SET_ELEC_CURRENT] == 1 else 0, axis=1)

    df.loc[df[SET_GRID_DIST_PLANNED] < 10, SET_ELEC_FUTURE] = pre_elec(df.loc[df[SET_GRID_DIST_PLANNED] < 10],
                                                                       grid_lcoes.to_dict())

    df[SET_LCOE_GRID] = 99
    df[SET_LCOE_GRID] = df.apply(lambda row: grid_price if row[SET_ELEC_FUTURE] == 1 else 99, axis=1)

    df[SET_LCOE_GRID], df[SET_MIN_GRID_DIST] = elec_direct(df, grid_lcoes.to_dict(),
                                                           existing_grid_cost_ratio, max_dist)

    return df


def results_columns(df, scenario, grid_btp, num_people_per_hh, diesel_price, grid_price,
                    transmission_losses, grid_capacity_investment, project_life):
    def res_investment_cost(row):
        min_tech = row[SET_MINIMUM_OVERALL]
        if min_tech == SET_LCOE_SA_DIESEL:
            return get_sa_diesel_lcoe(row[SET_NEW_CONNECTIONS], scenario, num_people_per_hh, True, diesel_price, project_life=project_life)
        elif min_tech == SET_LCOE_SA_PV:
            return get_sa_pv_lcoe(row[SET_NEW_CONNECTIONS], scenario, num_people_per_hh, True, row[SET_GHI], project_life=project_life)
        elif min_tech == SET_LCOE_MG_WIND:
            return get_mg_wind_lcoe(row[SET_NEW_CONNECTIONS], scenario, num_people_per_hh, True, row[SET_WINDCF], project_life=project_life)
        elif min_tech == SET_LCOE_MG_DIESEL:
            return get_mg_diesel_lcoe(row[SET_NEW_CONNECTIONS], scenario, num_people_per_hh, True, diesel_price, project_life=project_life)
        elif min_tech == SET_LCOE_MG_PV:
            return get_mg_pv_lcoe(row[SET_NEW_CONNECTIONS], scenario, num_people_per_hh, True, row[SET_GHI], project_life=project_life)
        elif min_tech == SET_LCOE_MG_HYDRO:
            return get_mg_hydro_lcoe(row[SET_NEW_CONNECTIONS], scenario, num_people_per_hh, True, row[SET_HYDRO_DIST], project_life=project_life)
        elif min_tech == SET_LCOE_GRID:
            return get_grid_lcoe(row[SET_NEW_CONNECTIONS], scenario, num_people_per_hh, True, transmission_losses,
                                 grid_btp, grid_price, grid_capacity_investment, row[SET_MIN_GRID_DIST], project_life=project_life)
        else:
            raise ValueError('A technology has not been accounted for in res_investment_cost()')

    df[SET_MINIMUM_OVERALL] = df[[SET_LCOE_GRID, SET_LCOE_SA_DIESEL, SET_LCOE_SA_PV, SET_LCOE_MG_WIND,
                                  SET_LCOE_MG_DIESEL, SET_LCOE_MG_PV, SET_LCOE_MG_HYDRO]].T.idxmin()

    df[SET_MINIMUM_OVERALL_LCOE] = df.apply(lambda row: (row[row[SET_MINIMUM_OVERALL]]), axis=1)

    codes = {SET_LCOE_GRID: 1, SET_LCOE_MG_HYDRO: 7, SET_LCOE_MG_WIND: 6, SET_LCOE_MG_PV: 5,
             SET_LCOE_MG_DIESEL: 4, SET_LCOE_SA_DIESEL: 2, SET_LCOE_SA_PV: 3}
    df.loc[df[SET_MINIMUM_OVERALL] == SET_LCOE_GRID, SET_MINIMUM_OVERALL_CODE] = codes[SET_LCOE_GRID]
    df.loc[df[SET_MINIMUM_OVERALL] == SET_LCOE_MG_HYDRO, SET_MINIMUM_OVERALL_CODE] = codes[SET_LCOE_MG_HYDRO]
    df.loc[df[SET_MINIMUM_OVERALL] == SET_LCOE_SA_PV, SET_MINIMUM_OVERALL_CODE] = codes[SET_LCOE_SA_PV]
    df.loc[df[SET_MINIMUM_OVERALL] == SET_LCOE_MG_WIND, SET_MINIMUM_OVERALL_CODE] = codes[SET_LCOE_MG_WIND]
    df.loc[df[SET_MINIMUM_OVERALL] == SET_LCOE_MG_PV, SET_MINIMUM_OVERALL_CODE] = codes[SET_LCOE_MG_PV]
    df.loc[df[SET_MINIMUM_OVERALL] == SET_LCOE_MG_DIESEL, SET_MINIMUM_OVERALL_CODE] = codes[SET_LCOE_MG_DIESEL]
    df.loc[df[SET_MINIMUM_OVERALL] == SET_LCOE_SA_DIESEL, SET_MINIMUM_OVERALL_CODE] = codes[SET_LCOE_SA_DIESEL]

    df[SET_MINIMUM_CATEGORY] = df[SET_MINIMUM_OVERALL].str.extract('(sa|mg|grid)', expand=False)

    grid_vals = {'cf': 1.0, 'btp': grid_btp}
    mg_hydro_vals = {'cf': 0.5, 'btp': 1.0}
    mg_pv_vals = {'btp': 0.9}
    mg_wind_vals = {'btp': 0.75}
    mg_diesel_vals = {'cf': 0.7, 'btp': 0.5}
    sa_diesel_vals = {'cf': 0.7, 'btp': 0.5}
    sa_pv_vals = {'btp': 0.9}

    df.loc[df[SET_MINIMUM_OVERALL] == SET_LCOE_GRID, SET_NEW_CAPACITY] = \
        (df[SET_NEW_CONNECTIONS] * scenario / num_people_per_hh) / (HOURS_PER_YEAR * grid_vals['cf'] *
                                                                    grid_vals['btp'])
    df.loc[df[SET_MINIMUM_OVERALL] == SET_LCOE_MG_HYDRO, SET_NEW_CAPACITY] = \
        (df[SET_NEW_CONNECTIONS] * scenario / num_people_per_hh) / (HOURS_PER_YEAR * mg_hydro_vals['cf'] *
                                                                    mg_hydro_vals['btp'])
    df.loc[df[SET_MINIMUM_OVERALL] == SET_LCOE_MG_PV, SET_NEW_CAPACITY] = \
        (df[SET_NEW_CONNECTIONS] * scenario / num_people_per_hh) / (HOURS_PER_YEAR * (df[SET_GHI] / HOURS_PER_YEAR) *
                                                                    mg_pv_vals['btp'])
    df.loc[df[SET_MINIMUM_OVERALL] == SET_LCOE_MG_WIND, SET_NEW_CAPACITY] = \
        (df[SET_NEW_CONNECTIONS] * scenario / num_people_per_hh) / (HOURS_PER_YEAR * df[SET_WINDCF] *
                                                                    mg_wind_vals['btp'])
    df.loc[df[SET_MINIMUM_OVERALL] == SET_LCOE_MG_DIESEL, SET_NEW_CAPACITY] = \
        (df[SET_NEW_CONNECTIONS] * scenario / num_people_per_hh) / (HOURS_PER_YEAR * mg_diesel_vals['cf'] *
                                                                    mg_diesel_vals['btp'])
    df.loc[df[SET_MINIMUM_OVERALL] == SET_LCOE_SA_DIESEL, SET_NEW_CAPACITY] = \
        (df[SET_NEW_CONNECTIONS] * scenario / num_people_per_hh) / (HOURS_PER_YEAR * sa_diesel_vals['cf'] *
                                                                    sa_diesel_vals['btp'])
    df.loc[df[SET_MINIMUM_OVERALL] == SET_LCOE_SA_PV, SET_NEW_CAPACITY] = \
        (df[SET_NEW_CONNECTIONS] * scenario / num_people_per_hh) / (HOURS_PER_YEAR * (df[SET_GHI] / HOURS_PER_YEAR) *
                                                                    sa_pv_vals['btp'])

    df[SET_INVESTMENT_COST] = df.apply(res_investment_cost, axis=1)
    return df


def techs_only(df, diesel_price, scenario, num_people_per_hh, mg_hydro_capital_cost, mg_pv_capital_cost,
               mg_wind_capital_cost, mg_diesel_capital_cost, sa_diesel_capital_cost, sa_pv_capital_cost, project_life):
    # Prepare MG_DIESEL
    # Pp = p_lcoe + (2*p_d*consumption*time/volume)*(1/mu)*(1/LHVd)
    consumption_mg_diesel = 33.7
    volume_mg_diesel = 15000
    mu_mg_diesel = 0.3

    # Prepare SA_DIESEL
    # Pp = (p_d + 2*p_d*consumption*time/volume)*(1/mu)*(1/LHVd) + p_om + p_c
    consumption_sa_diesel = 14  # (l/h) truck consumption per hour
    volume_sa_diesel = 300  # (l) volume of truck
    mu_sa_diesel = 0.3  # (kWhth/kWhel) gen efficiency
    p_om_sa_diesel = 0.01  # (USD/kWh) operation, maintenance and amortization

    df[SET_LCOE_MG_HYDRO] = df.apply(
        lambda row: get_mg_hydro_lcoe(row[SET_POP_FUTURE], scenario, num_people_per_hh, False, row[SET_HYDRO_DIST],
                                      mg_hydro_capital_cost, project_life)
        if row[SET_HYDRO_DIST] < 5 else 99, axis=1)

    df[SET_LCOE_MG_PV] = df.apply(
        lambda row: get_mg_pv_lcoe(row[SET_POP_FUTURE], scenario, num_people_per_hh, False, row[SET_GHI],
                                   mg_pv_capital_cost, project_life)
        if (row[SET_SOLAR_RESTRICTION] == 1 and row[SET_GHI] > 1000) else 99,
        axis=1)

    df[SET_LCOE_MG_WIND] = df.apply(
        lambda row: get_mg_wind_lcoe(row[SET_POP_FUTURE], scenario, num_people_per_hh, False, row[SET_WINDCF],
                                     mg_wind_capital_cost, project_life)
        if row[SET_WINDCF] > 0.1 else 99
        , axis=1)

    df[SET_LCOE_MG_DIESEL] = df.apply(
        lambda row:
        get_mg_diesel_lcoe(row[SET_POP_FUTURE], scenario, num_people_per_hh, False, diesel_price,
                           mg_diesel_capital_cost, project_life) +
        (2 * diesel_price * consumption_mg_diesel * row[SET_TRAVEL_HOURS] / volume_mg_diesel) *
        (1 / mu_mg_diesel) * (1 / LHV_DIESEL),
        axis=1)

    df[SET_LCOE_SA_DIESEL] = df.apply(
        lambda row:
        (diesel_price + 2 * diesel_price * consumption_sa_diesel * row[SET_TRAVEL_HOURS] / volume_sa_diesel) *
        (1 / mu_sa_diesel) * (1 / LHV_DIESEL) + p_om_sa_diesel + get_sa_diesel_lcoe(row[SET_POP_FUTURE], scenario,
                                                                                    num_people_per_hh, False,
                                                                                    diesel_price,
                                                                                    sa_diesel_capital_cost,
                                                                                    project_life),
        axis=1)

    df[SET_LCOE_SA_PV] = df.apply(
        lambda row: get_sa_pv_lcoe(row[SET_POP_FUTURE], scenario, num_people_per_hh, False, row[SET_GHI],
                                   sa_pv_capital_cost, project_life)
        if row[SET_GHI] > 1000 else 99,
        axis=1)

    df[SET_MINIMUM_TECH] = df[[SET_LCOE_SA_DIESEL, SET_LCOE_SA_PV, SET_LCOE_MG_WIND,
                               SET_LCOE_MG_DIESEL, SET_LCOE_MG_PV, SET_LCOE_MG_HYDRO]].T.idxmin()

    df[SET_MINIMUM_TECH_LCOE] = df.apply(lambda row: (row[row[SET_MINIMUM_TECH]]), axis=1)

    return df