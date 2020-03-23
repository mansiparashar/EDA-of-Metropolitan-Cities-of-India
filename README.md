
## CONTRASTING AND COMPARING THE METROPOLITANS OF INDIA

##### A Capstone Project completed as per the requirements of the [IBM Data Science Professional  Certificate](https://www.youracclaim.com/badges/7fa828dd-4931-4da3-ba74-e743f7a8c4c6/public_url "IBM Data Science Professional  Certificate")

------------

##### 1.INTRODUCTION
###### 1.1 BACKGROUND
India is the 7th largest country in the world, and the 2
nd most populated country. India is a developing State and has 4 metropolitan cities so to speak of, Delhi which is
the capital city, Mumbai(formerly Bombay), Chennai(formerly Madras) and
Kolkata(formerly Calcutta). In recent years, India has seen a tremendous boost in job opportunities and the general
quality of life in these metros, leading to a large rate of migration to the aforementioned
cities. These 4 cities belong to different parts of the country, and hence vary significantly in the
type of life they offer. 
###### 1.2 Problem Statement
Aim is to analyse the 4 metropolitans of India ; explore the neighborhoods with relevant
data and state the final findings in terms of the various venues and amenities provided
at each city. Contrast and compare this data retrieved for the 4 cities on the basis of different clusters
formed by the neighborhoods and the services/amenities these clusters have to offer, and how they line with the interest of the stakeholders. Also, compare the metropolitans on the basis of different unique amenities provided by
them and the corresponding quantity. Finally, identify the different regions in each metropolitan based on shared common
venues to locate possible choices for potential residents
### 2. DATA 
We work on 3 sets of data :- <br><br>
**2.1** Data containing the neighborhoods, pincodes,latitude, longitude and for cities in India.This data is accessed from the official government of India
site,https://www.india.gov.in/ which serves as a national portal for accessing different
data concerning the country of India. To access the data set, we need to fill in a form declaring we’re using this data for
academic purposes. Unfortunately, all the latitude and longitude values were NaNs.<br><br>
**2.2** Furthermore, we found a data set containing Indian pincodes and their latitude
and longitudes from the same site. <br><br>
**2.3** Based on the latitudes and longitudes in our database, we use the FourSquare API
to access data about the neighborhood venues required for this study. The Foursquare Places API provides location based experiences with diverse
information about venues, users, photos, and check-ins. The API supports real time
access to places. Additionally, Foursquare allows developers to build audience segments for analysis and
measurement. JSON is the preferred response format which needs to be converted into the required
dataframe
Accessing the API would require us to make an account on the Developer’s Portal on the
foursquare website,https://developer.foursquare.com/







