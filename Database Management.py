###############################Part 1##################################

'''
Create an SQLite database and import the data into a table named
“CarSharing”. Create a backup table and copy the whole table into it.
'''

import pandas as pd

# Load the CSV file to understand its structure
csv_file_path = 'CarSharing.csv'
car_sharing_df = pd.read_csv(csv_file_path)

# Display the first few rows of the dataframe to understand its structure and data types
print(car_sharing_df.head(), car_sharing_df.dtypes)

import sqlite3

# Create a new SQLite database
db_path = 'CarSharingDB.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create the CarSharing table
create_table_query = '''
CREATE TABLE CarSharing (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    season TEXT,
    holiday TEXT,
    workingday TEXT,
    weather TEXT,
    temp REAL,
    temp_feel REAL,
    humidity REAL,
    windspeed REAL,
    demand REAL
)
'''
cursor.execute(create_table_query)

# Import data into the CarSharing table
car_sharing_df.to_sql('CarSharing', conn, if_exists='replace', index=False)

# Create a backup table with the same structure as CarSharing
cursor.execute('CREATE TABLE CarSharingBackup AS SELECT * FROM CarSharing WHERE 1=0')

# Copy all data from CarSharing to CarSharingBackup
cursor.execute('INSERT INTO CarSharingBackup SELECT * FROM CarSharing')

# Commit the changes and close the connection
conn.commit()

# Verify by counting the rows in both tables
row_count_carsharing = cursor.execute('SELECT COUNT(*) FROM CarSharing').fetchone()[0]
row_count_backup = cursor.execute('SELECT COUNT(*) FROM CarSharingBackup').fetchone()[0]

conn.close()

print(row_count_carsharing, row_count_backup)


########################################Part 2#######################################

'''
Add a column to the CarSharing table named “temp_category”. This
column should contain three string values. If the “feels-like”
temperature is less than 10 then the corresponding value in the
temp_category column should be “Cold”, if the feels-like temperature is
between 10 and 25, the value should be “Mild”, and if the feels-like
temperature is greater than 25, then the value should be “Hot”.
'''

# Reopen the connection to the SQLite database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Step 1: Add the new column "temp_category" to the "CarSharing" table
cursor.execute('ALTER TABLE CarSharing ADD COLUMN temp_category TEXT')

# Step 2: Update the "temp_category" column based on "temp_feel" values
cursor.execute('UPDATE CarSharing SET temp_category = "Cold" WHERE temp_feel < 10')
cursor.execute('UPDATE CarSharing SET temp_category = "Mild" WHERE temp_feel >= 10 AND temp_feel <= 25')
cursor.execute('UPDATE CarSharing SET temp_category = "Hot" WHERE temp_feel > 25')

# Commit the changes
conn.commit()

# Step 3: Verification - Select a few rows to check the "temp_category" values
verification_query = 'SELECT id, temp_feel, temp_category FROM CarSharing LIMIT 10'
verification_results = cursor.execute(verification_query).fetchall()

conn.close()

print(verification_results)

#################################Part 3####################################

'''
Create another table named “temperature” by selecting the temp,
temp_feel, and temp_category columns. Then drop the temp and temp_feel
columns from the CarSharing table.
'''

# Reopen the connection to the SQLite database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Step 1: Create the "temperature" table
cursor.execute('CREATE TABLE temperature AS SELECT id, temp, temp_feel, temp_category FROM CarSharing')

# Step 2: Remove the "temp" and "temp_feel" columns from the "CarSharing" table
# 2.1 Create a temporary table excluding the "temp" and "temp_feel" columns
cursor.execute('''
CREATE TABLE CarSharing_temp AS
SELECT id, timestamp, season, holiday, workingday, weather, humidity, windspeed, demand, temp_category
FROM CarSharing
''')

# 2.2 Drop the original "CarSharing" table
cursor.execute('DROP TABLE CarSharing')

# 2.3 Rename the temporary table to "CarSharing"
cursor.execute('ALTER TABLE CarSharing_temp RENAME TO CarSharing')

# Commit the changes
conn.commit()

# Step 3: Verification
# 3.1 Verify the "temperature" table structure
temperature_table_structure = cursor.execute('PRAGMA table_info(temperature)').fetchall()

# 3.2 Verify the modified "CarSharing" table structure
carsharing_table_structure = cursor.execute('PRAGMA table_info(CarSharing)').fetchall()

conn.close()

print(temperature_table_structure, carsharing_table_structure)

###################################Part 4##################################

'''
Find the distinct values of the weather column and assign a number
to each value. Add another column named “weather_code” to the table
containing each row’s assigned weather code.
'''

# Reopen the connection to the SQLite database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Step 1: Identify distinct weather values
distinct_weather = cursor.execute('SELECT DISTINCT weather FROM CarSharing').fetchall()
distinct_weather = [weather[0] for weather in distinct_weather]

# Step 2: Assign a unique number to each weather condition
weather_codes = {weather: code for code, weather in enumerate(distinct_weather, start=1)}

# Step 3: Add the "weather_code" column to the "CarSharing" table
cursor.execute('ALTER TABLE CarSharing ADD COLUMN weather_code INTEGER')

# Step 4: Update the "weather_code" column based on the weather condition
for weather, code in weather_codes.items():
    cursor.execute('UPDATE CarSharing SET weather_code = ? WHERE weather = ?', (code, weather))

# Commit the changes
conn.commit()

# Step 5: Verification - Select a few rows to check the "weather" and "weather_code" values
verification_query = 'SELECT id, weather, weather_code FROM CarSharing LIMIT 10'
verification_results = cursor.execute(verification_query).fetchall()

conn.close()

print(weather_codes, verification_results)

##################################Part 5#####################################

'''
Create a table called “weather” and copy the columns “weather” and
“weather_code” to this table. Then drop the weather column from the
CarSharing table.
'''

# Reopen the connection to the SQLite database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Step 1: Create the "weather" table with unique pairs of "weather" conditions and "weather_code"
cursor.execute('CREATE TABLE weather AS SELECT DISTINCT weather, weather_code FROM CarSharing')

# Step 3: Remove the "weather" column from the "CarSharing" table using the workaround
# 3.1 Create a temporary table without the "weather" column
cursor.execute('''
CREATE TABLE CarSharing_temp AS
SELECT id, timestamp, season, holiday, workingday, humidity, windspeed, demand, temp_category, weather_code
FROM CarSharing
''')

# 3.2 Drop the original "CarSharing" table
cursor.execute('DROP TABLE CarSharing')

# 3.3 Rename the temporary table to "CarSharing"
cursor.execute('ALTER TABLE CarSharing_temp RENAME TO CarSharing')

# Commit the changes
conn.commit()

# Step 4: Verification
# 4.1 Verify the "weather" table structure
weather_table_structure = cursor.execute('PRAGMA table_info(weather)').fetchall()

# 4.2 Verify the modified "CarSharing" table structure
carsharing_table_structure = cursor.execute('PRAGMA table_info(CarSharing)').fetchall()

conn.close()

print(weather_table_structure, carsharing_table_structure)

#####################################Part 6#####################################

'''
Create a table called "time" with four columns containing each row’s
timestamp, hour, weekday name, and month name (Hint: you can use the
strftime() function for this purpose).
'''

# Reopen the connection to the SQLite database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Step 1 & 2: Create the "time" table with timestamp, hour, weekday name, and month name
# SQLite treats Monday as 1, hence the adjustment in weekday calculation to match common expectations
cursor.execute('''
CREATE TABLE time AS
SELECT 
    timestamp,
    CAST(strftime('%H', timestamp) AS INTEGER) AS hour,
    CASE CAST(strftime('%w', timestamp) AS INTEGER)
        WHEN 0 THEN 'Sunday'
        WHEN 1 THEN 'Monday'
        WHEN 2 THEN 'Tuesday'
        WHEN 3 THEN 'Wednesday'
        WHEN 4 THEN 'Thursday'
        WHEN 5 THEN 'Friday'
        WHEN 6 THEN 'Saturday'
    END AS weekday,
    CASE CAST(strftime('%m', timestamp) AS INTEGER)
        WHEN 1 THEN 'January'
        WHEN 2 THEN 'February'
        WHEN 3 THEN 'March'
        WHEN 4 THEN 'April'
        WHEN 5 THEN 'May'
        WHEN 6 THEN 'June'
        WHEN 7 THEN 'July'
        WHEN 8 THEN 'August'
        WHEN 9 THEN 'September'
        WHEN 10 THEN 'October'
        WHEN 11 THEN 'November'
        WHEN 12 THEN 'December'
    END AS month
FROM CarSharing
''')

# Commit the changes
conn.commit()

# Step 4: Verification - Select a few rows to check the "time" table structure and data
verification_query = 'SELECT * FROM time LIMIT 10'
verification_results = cursor.execute(verification_query).fetchall()

conn.close()

print(verification_results)


#####################################Part 7###################################

'''
Assume it’s the first day you have started working at this company
and your boss Linda sends you an email as follows: “Hello, welcome to
the team. I hope you enjoy working at this company. Could you please
give me a report containing the following information:
  
a) Please tell me which date and time we had the highest demand rate in 2017.
     
b) Give me a table containing the name of the weekday, month, and season
in which we had the highest and lowest average demand rates throughout
2017. Please include the calculated average demand values as well.
 
c) For the weekday selected in (b), please give me a table showing the
average demand rate we had at different hours of that weekday throughout
2017. Please sort the results in descending order based on the average
demand rates.

d) Please tell me what the weather was like in 2017. Was it mostly cold,
mild, or hot? Which weather condition (shown in the weather column) was
the most prevalent in 2017? What was the average, highest, and lowest
wind speed and humidity for each month in 2017? Please organise this
information in two tables for the wind speed and humidity. Please also
give me a table showing the average demand rate for each cold, mild, and
hot weather in 2017 sorted in descending order based on their average
demand rates.

e) Give me another table showing the information requested in (d) for
the month we had the highest average demand rate in 2017 so that I can
compare it with other months.
'''

# a

# Reopen the connection to the SQLite database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Part (a): Highest demand rate date and time in 2017
highest_demand_query = '''
SELECT timestamp, MAX(demand) as max_demand
FROM CarSharing
WHERE strftime('%Y', timestamp) = '2017'
'''
highest_demand_result = cursor.execute(highest_demand_query).fetchone()

print(highest_demand_result)

# b

# Part (b): Weekday, Month, and Season with Highest and Lowest Average Demand Rates in 2017
average_demand_query = '''
SELECT t.weekday, t.month, c.season, AVG(c.demand) as avg_demand
FROM CarSharing c
JOIN time t ON c.timestamp = t.timestamp
WHERE strftime('%Y', c.timestamp) = '2017'
GROUP BY t.weekday, t.month, c.season
ORDER BY avg_demand DESC
'''

average_demand_results = cursor.execute(average_demand_query).fetchall()

# Extracting the rows with the highest and lowest average demand
highest_average_demand = average_demand_results[0]
lowest_average_demand = average_demand_results[-1]

print(highest_average_demand, lowest_average_demand)

# c

# Part (c): Average Demand Rate at Different Hours for Sunday throughout 2017
average_demand_by_hour_query = '''
SELECT t.hour, AVG(c.demand) as avg_demand
FROM CarSharing c
JOIN time t ON c.timestamp = t.timestamp
WHERE strftime('%Y', c.timestamp) = '2017' AND t.weekday = 'Sunday'
GROUP BY t.hour
ORDER BY avg_demand DESC
'''

average_demand_by_hour_results = cursor.execute(average_demand_by_hour_query).fetchall()

print(average_demand_by_hour_results)

# d

# Part (d) Weather overview in 2017: Predominant temperature category and prevalent weather condition
temperature_category_query = '''
SELECT temp_category, COUNT(temp_category) as count
FROM CarSharing
WHERE strftime('%Y', timestamp) = '2017'
GROUP BY temp_category
ORDER BY count DESC
LIMIT 1
'''

most_prevalent_weather_condition_query = '''
SELECT weather, COUNT(weather) as count
FROM weather
JOIN CarSharing ON weather.weather_code = CarSharing.weather_code
WHERE strftime('%Y', CarSharing.timestamp) = '2017'
GROUP BY weather
ORDER BY count DESC
LIMIT 1
'''

# Average, highest, and lowest wind speed and humidity for each month in 2017
wind_speed_humidity_query = '''
SELECT 
    strftime('%m', timestamp) as month,
    AVG(windspeed) as avg_windspeed, MAX(windspeed) as max_windspeed, MIN(windspeed) as min_windspeed,
    AVG(humidity) as avg_humidity, MAX(humidity) as max_humidity, MIN(humidity) as min_humidity
FROM CarSharing
WHERE strftime('%Y', timestamp) = '2017'
GROUP BY month
ORDER BY month
'''

# Average demand rate for each weather condition in 2017
average_demand_by_weather_condition_query = '''
SELECT temp_category, AVG(demand) as avg_demand
FROM CarSharing
WHERE strftime('%Y', timestamp) = '2017'
GROUP BY temp_category
ORDER BY avg_demand DESC
'''

# Execute queries
predominant_temperature_category = cursor.execute(temperature_category_query).fetchone()
most_prevalent_weather_condition = cursor.execute(most_prevalent_weather_condition_query).fetchone()
wind_speed_humidity_stats = cursor.execute(wind_speed_humidity_query).fetchall()
average_demand_by_weather_condition = cursor.execute(average_demand_by_weather_condition_query).fetchall()

print(predominant_temperature_category)
print(most_prevalent_weather_condition)
print(wind_speed_humidity_stats)
print(average_demand_by_weather_condition)

# e

# Reopen the connection to the SQLite database for the targeted analysis
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Step 1: Identify the month with the highest average demand rate in 2017
highest_avg_demand_month_query = '''
SELECT strftime('%m', timestamp) AS month, AVG(demand) AS avg_demand
FROM CarSharing
WHERE strftime('%Y', timestamp) = '2017'
GROUP BY month
ORDER BY avg_demand DESC
LIMIT 1
'''
highest_avg_demand_month = cursor.execute(highest_avg_demand_month_query).fetchone()[0]

# Step 2: Retrieve wind speed and humidity statistics for the identified month
wind_speed_humidity_for_month_query = '''
SELECT 
    AVG(windspeed) AS avg_windspeed, MAX(windspeed) AS max_windspeed, MIN(windspeed) AS min_windspeed,
    AVG(humidity) AS avg_humidity, MAX(humidity) AS max_humidity, MIN(humidity) AS min_humidity
FROM CarSharing
WHERE strftime('%Y', timestamp) = '2017' AND strftime('%m', timestamp) = ?
'''
wind_speed_humidity_for_month_stats = cursor.execute(wind_speed_humidity_for_month_query, (highest_avg_demand_month,)).fetchall()

# Step 3: Compare average demand rates by temperature category for the identified month
average_demand_by_temp_category_for_month_query = '''
SELECT temp_category, AVG(demand) AS avg_demand
FROM CarSharing
WHERE strftime('%Y', timestamp) = '2017' AND strftime('%m', timestamp) = ?
GROUP BY temp_category
ORDER BY avg_demand DESC
'''
average_demand_by_temp_category_for_month = cursor.execute(average_demand_by_temp_category_for_month_query, (highest_avg_demand_month,)).fetchall()

# Close the connection
conn.close()

print(highest_avg_demand_month, wind_speed_humidity_for_month_stats, average_demand_by_temp_category_for_month)
