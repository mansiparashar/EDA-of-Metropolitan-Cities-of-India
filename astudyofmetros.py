#!/usr/bin/env python
# coding: utf-8

# ### Data colection and cleaning

# In[ ]:


#installing necessary packages
get_ipython().system('conda install -c conda-forge geopy --yes')
get_ipython().system('conda install -c conda-forge folium=0.5.0 --yes')


# In[1]:


#importing libraries
import pandas as pd
import numpy as np
import requests
from geopy.geocoders import Nominatim
from IPython.display import Image 
from IPython.core.display import HTML 
from pandas.io.json import json_normalize
import matplotlib.cm as cm
import matplotlib.colors as colors
from sklearn.cluster import KMeans
import folium


# In[2]:


#reading dataset
dataset=pd.read_csv(r'D:\Downloads\all_india_PO_list_without_APS_offices_ver2_lat_long.csv')


# In[3]:


#cleaning up irrelevant columns
df1=dataset[['officename' ,'pincode', 'regionname']]
df1.rename(columns = {"officename": "neighborhood"},inplace = True) 
df = df1.groupby(['pincode', 'regionname'], sort = False).agg(','.join)
df.reset_index(inplace=True)
df.head()


# In[4]:


#reading dataset contaaining location data
ind=pd.read_csv(r'C:\Users\91984\Desktop\in.csv')
ind.rename(columns = {"key": "pincode","place_name": "neighborhood","admin_name1": "city"},inplace = True) 
del ind['neighborhood']
del ind['city']
del ind['accuracy']
ind.head()


# In[5]:


#merging the 2 to get final dataset
d=pd.merge(ind, df, on='pincode')
d.head()


# ### Obtain individual datasets for each city

# In[6]:


delhi=d.loc[d['regionname'] == 'Delhi']
delhi.reset_index()
delhi


# In[7]:


#mumbai
mumbai=d.loc[d['regionname'] == 'Mumbai']
mumbai.reset_index(inplace=True)
del mumbai['index']
mumbai


# In[8]:


#kolkata
kolkata=d.loc[d['regionname'] == 'Calcutta']
kolkata.reset_index(inplace=True)
del kolkata['index']
kolkata


# In[9]:


#chennai
Chennai=d.loc[d['regionname'] == 'Chennai Region']
Chennai.reset_index(inplace=True)
del Chennai['index']
Chennai


# ### choose a non metropolitan city for comparison's sake

# In[20]:


#kanpur
kanpur=d.loc[d['regionname'] == 'Kanpur']
kanpur.reset_index(inplace=True)
del kanpur['index']
kanpur


# ### Visualize each

# In[21]:


#kanpur
from geopy.geocoders import Nominatim
import matplotlib.cm as cm
import matplotlib.colors as colors
from sklearn.cluster import KMeans
import folium
#address = 'Mumbai'
#geolocator = Nominatim(user_agent="Toronto_explorer")
#location = geolocator.geocode(address)
#latitude = location.latitude
#longitude = location.longitude

Tomap = folium.Map(location=[26.4750,80.3083], zoom_start=10)

for lat, lng, city, neighborhood in zip(kanpur['latitude'], kanpur['longitude'],kanpur['regionname'], kanpur['neighborhood']):
    label = '{}, {}'.format(neighborhood,city )
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='green',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(Tomap)  
    
Tomap


# In[10]:


#chennai
from geopy.geocoders import Nominatim
import matplotlib.cm as cm
import matplotlib.colors as colors
from sklearn.cluster import KMeans
import folium
#address = 'Mumbai'
#geolocator = Nominatim(user_agent="Toronto_explorer")
#location = geolocator.geocode(address)
#latitude = location.latitude
#longitude = location.longitude

Tomap = folium.Map(location=[13.0656,80.2672], zoom_start=10)

for lat, lng, city, neighborhood in zip(Chennai['latitude'], Chennai['longitude'], 
                                           Chennai['regionname'], Chennai['neighborhood']):
    label = '{}, {}'.format(neighborhood,city )
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(Tomap)  
    
Tomap


# In[11]:


Tomap = folium.Map(location=[19.0167,72.85], zoom_start=10)

for lat, lng, city, neighborhood in zip(mumbai['latitude'], mumbai['longitude'], 
                                           mumbai['regionname'], mumbai['neighborhood']):
    label = '{}, {}'.format(neighborhood,city )
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='red',
        fill_opacity=0.7,
        parse_html=False).add_to(Tomap)  
    
Tomap


# In[12]:


#delhi
dmap = folium.Map(location=[28.6333,77.2167], zoom_start=10)

for lat, lng, city, neighborhood in zip(delhi['latitude'], delhi['longitude'], 
                                           delhi['regionname'],delhi['neighborhood']):
    label = '{}, {}'.format(neighborhood,city )
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='#69527E',
        fill=True,
        fill_color='#DCD5F6',
        fill_opacity=0.7,
        parse_html=False).add_to(dmap)  
    
dmap


# In[13]:


dmap = folium.Map(location=[22.5690,88.3697], zoom_start=7)

for lat, lng, city, neighborhood in zip(kolkata['latitude'], kolkata['longitude'], 
                                           kolkata['regionname'],kolkata['neighborhood']):
    label = '{}, {}'.format(neighborhood,city )
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=2,
        popup=label,
        color='#D9364E',
        fill=True,
        fill_color='#D9364E',
        fill_opacity=0.7,
        parse_html=False).add_to(dmap)  
    
dmap


# In[14]:


#segmentation
#define 4square credentials
CLIENT_ID = 'OQYWSZCAO25AV4Z3KPAZQXOKTGGAIQBJHJXSLLXOFG3I0DRW' # your Foursquare ID
CLIENT_SECRET = 'LRONLXGCQQFSHIBIBN4Z3X2WT0CCXUZFRREYSXO1G1QGXM4H' # your Foursquare Secret
VERSION = '20180605'


# In[15]:


#create a function to interact with the foursquare  API to get venue data
LIMIT = 7
def getNearbyVenues(names, latitudes, longitudes, radius=1000):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[16]:


delhi_venues=getNearbyVenues(names=delhi['neighborhood'],
                                   latitudes=delhi['latitude'],
                                   longitudes=delhi['longitude']
                                  )


# In[22]:


d_areas=delhi_venues['Venue Category'].nunique()
delhi_venues['Venue Category'].value_counts()


# In[24]:


import matplotlib.pyplot as plt
delhi_venues['Venue Category'].value_counts().plot(kind='barh', figsize=(25, 15))
plt.ylabel("Venues", fontsize=20)
plt.xlabel("Number", fontsize=14)
plt.title("Different venues in Delhi",fontsize=19);


# In[25]:


#get dummy values indication numbers of each category
#one hot encoding
d_onehot = pd.get_dummies(delhi_venues[['Venue Category']], prefix="", prefix_sep="")
# add neighborhood column back to dataframe
d_onehot['neighborhood'] =delhi_venues['Neighborhood'] 
fixed_columns = [d_onehot.columns[-1]] + list(d_onehot.columns[:-1])
d_onehot = d_onehot[fixed_columns]
d_onehot.head()


# In[45]:


#with 5 common venues
delhi_neighbourhood= d_onehot.groupby('neighborhood').mean().reset_index()
delhi_neighbourhood


# In[46]:


num_top_venues = 5

for hood in delhi_neighbourhood['neighborhood']:
    print("----"+hood+"----")
    temp = delhi_neighbourhood[delhi_neighbourhood['neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','frequency']
    temp = temp.iloc[1:]
    temp['frequency'] = temp['frequency'].astype(float)
    temp = temp.round({'frequency': 2})
    print(temp.sort_values('frequency', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# In[47]:


#convert to dataframe
#sort in descending

def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[48]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['neighborhood'] = delhi_neighbourhood['neighborhood']

for ind in np.arange(delhi_neighbourhood.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(delhi_neighbourhood.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# In[49]:


#k means clustering on delhi_neighborhood

# set number of clusters
kclusters = 5
#drop the non numeric column
d_cluster = delhi_neighbourhood.drop('neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(d_cluster)

# check cluster labels generated for each row in the dataframe

labels=kmeans.labels_
labels.dtype
neighborhoods_venues_sorted['labels']=labels
#neighborhoods_venues_sorted.dtypes
neighborhoods_venues_sorted


# In[50]:


d_merged = delhi
# merge to add latitude/longitude for each neighborhood
d_merged = d_merged.join(neighborhoods_venues_sorted.set_index('neighborhood'), on='neighborhood')
#d_merged["labels"] = d_merged["labels"].astype(int)
#d_merged.shape
d_merged.dropna(axis=0,inplace=True)
d_merged["labels"] = d_merged["labels"].astype(int)
d_merged


# In[51]:


#visualise

# create map
map_clusters = folium.Map(location=[28.6333,77.2167], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(d_merged['latitude'], d_merged['longitude'],d_merged['neighborhood'], d_merged['labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# In[52]:


mumbai_Venues = getNearbyVenues(names=mumbai['neighborhood'],
                                   latitudes=mumbai['latitude'],
                                   longitudes=mumbai['longitude']
                                  )



# In[53]:


m_areas=mumbai_Venues['Venue Category'].nunique()
mumbai_Venues['Venue Category'].value_counts()


# In[54]:


mumbai_Venues['Venue Category'].value_counts().plot(kind='barh', figsize=(25, 15))
plt.ylabel("Venues", fontsize=20)
plt.xlabel("Number", fontsize=14)
plt.title("Different venues in Mumbai",fontsize=19);


# In[55]:


mumbai_onehot = pd.get_dummies(mumbai_Venues[['Venue Category']], prefix="", prefix_sep="")
# add neighborhood column back to dataframe
mumbai_onehot['neighborhood'] =mumbai_Venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [mumbai_onehot.columns[-1]] + list(mumbai_onehot.columns[:-1])
mumbai_onehot = mumbai_onehot[fixed_columns]
mumbai_onehot.head()


# In[56]:


m = mumbai_onehot.groupby('neighborhood').mean().reset_index()
num_top_venues = 5

for hood in m['neighborhood']:
    print("----"+hood+"----")
    temp = m[m['neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))


# In[ ]:





# In[57]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sortedm = pd.DataFrame(columns=columns)
neighborhoods_venues_sortedm['neighborhood'] = m['neighborhood']


# In[58]:


for ind in np.arange(m.shape[0]):
    neighborhoods_venues_sortedm.iloc[ind, 1:] = return_most_common_venues(m.iloc[ind, :], num_top_venues)

neighborhoods_venues_sortedm.head()
# set number of clusters
kclusters = 5

m_cluster = m.drop('neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(m_cluster)

# check cluster labels generated for each row in the dataframe
labels=kmeans.labels_
neighborhoods_venues_sortedm['labels']=labels
neighborhoods_venues_sortedm


# In[59]:



m_merged = mumbai
# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
m_merged = m_merged.join(neighborhoods_venues_sortedm.set_index('neighborhood'), on='neighborhood')


# In[60]:


m_merged.dropna(axis=0,inplace=True)
m_merged["labels"] = m_merged["labels"].astype(int)
m_merged.head()


# In[61]:


# create map
map_clusters = folium.Map(location=[19.0167,72.85], zoom_start=8)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(m_merged['latitude'], m_merged['longitude'],m_merged['neighborhood'], m_merged['labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# In[62]:


#kolkata
kolkata_Venues = getNearbyVenues(names=kolkata['neighborhood'],
                                   latitudes=kolkata['latitude'],
                                   longitudes=kolkata['longitude']
                                  )


# In[63]:


kolkata_onehot = pd.get_dummies(kolkata_Venues[['Venue Category']], prefix="", prefix_sep="")
# add neighborhood column back to dataframe
kolkata_onehot['neighborhood'] =kolkata_Venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [kolkata_onehot.columns[-1]] + list(kolkata_onehot.columns[:-1])
kolkata_onehot = kolkata_onehot[fixed_columns]
kolkata_onehot.head()


# In[65]:


k_areas=kolkata_Venues['Venue Category'].nunique()
kolkata_Venues['Venue Category'].value_counts()


# In[66]:


kolkata_Venues['Venue Category'].value_counts().plot(kind='barh', figsize=(25, 15))
plt.ylabel("Venues", fontsize=20)
plt.xlabel("Number", fontsize=14)
plt.title("Different venues in Kolkata",fontsize=19);


# In[67]:


kol = kolkata_onehot.groupby('neighborhood').mean().reset_index()
kol


# In[68]:


num_top_venues = 5

for hood in kol['neighborhood']:
    print("----"+hood+"----")
    temp = kol[kol['neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))


# In[69]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['neighborhood'] = kol['neighborhood']


# In[70]:


for ind in np.arange(kol.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(kol.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()
# set number of clusters
kclusters = 5

k_cluster = kol.drop('neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(k_cluster)

# check cluster labels generated for each row in the dataframe
labels=kmeans.labels_
neighborhoods_venues_sorted['labels']=labels
neighborhoods_venues_sorted


# In[71]:


k_merged = kolkata
k_merged = k_merged.join(neighborhoods_venues_sorted.set_index('neighborhood'), on='neighborhood')
# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood

k_merged.dropna(axis=0,inplace=True)
k_merged["labels"] = k_merged["labels"].astype(int)
k_merged


# In[72]:


# create map
map_clusters = folium.Map(location=[22.5690,88.3697], zoom_start=13)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(k_merged['latitude'], k_merged['longitude'],m_merged['neighborhood'], k_merged['labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# In[73]:


#chennai

chennai_Venues = getNearbyVenues(names=Chennai['neighborhood'],
                                   latitudes=Chennai['latitude'],
                                   longitudes=Chennai['longitude']
                                  )


# In[74]:


chennai_onehot = pd.get_dummies(chennai_Venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
chennai_onehot['neighborhood'] =chennai_Venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [chennai_onehot.columns[-1]] + list(chennai_onehot.columns[:-1])
chennai_onehot = chennai_onehot[fixed_columns]
chennai_onehot.head()


# In[75]:


c_areas=chennai_Venues['Venue Category'].nunique()
chennai_Venues['Venue Category'].value_counts()


# In[76]:


chennai_Venues['Venue Category'].value_counts().plot(kind='barh', figsize=(15, 15))
plt.ylabel("Venues", fontsize=20)
plt.xlabel("Number", fontsize=14)
plt.title("Different venues in chennai",fontsize=19);


# In[77]:


c = chennai_onehot.groupby('neighborhood').mean().reset_index()
c


# In[78]:


num_top_venues = 5

for hood in c['neighborhood']:
    print("----"+hood+"----")
    temp = c[c['neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# In[79]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['neighborhood'] = c['neighborhood']

for ind in np.arange(c.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(c.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# In[80]:


kclusters = 5

c_cluster = c.drop('neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(c_cluster)

# check cluster labels generated for each row in the dataframe
labels=kmeans.labels_
neighborhoods_venues_sorted['labels']=labels
neighborhoods_venues_sorted


# In[81]:


c_merged = Chennai

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
c_merged = c_merged.join(neighborhoods_venues_sorted.set_index('neighborhood'), on='neighborhood')
c_merged.dropna(axis=0,inplace=True)
c_merged["labels"] = c_merged["labels"].astype(int)
c_merged.head() # check the last columns!


# In[82]:


# create map
map_clusters = folium.Map(location=[13.0656,80.2672], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(c_merged['latitude'], c_merged['longitude'],c_merged['neighborhood'], c_merged['labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# In[26]:


#kanpur

kanpur_Venues = getNearbyVenues(names=kanpur['neighborhood'],
                                   latitudes=kanpur['latitude'],
                                   longitudes=kanpur['longitude']
                                  )


# In[27]:


kanpur_onehot = pd.get_dummies(kanpur_Venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
kanpur_onehot['neighborhood'] =kanpur_Venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [kanpur_onehot.columns[-1]] + list(kanpur_onehot.columns[:-1])
kanpur_onehot = kanpur_onehot[fixed_columns]
kanpur_onehot.head()


# In[28]:


kan_areas=kanpur_Venues['Venue Category'].nunique()
kanpur_Venues['Venue Category'].value_counts()


# In[29]:


kanpur_Venues['Venue Category'].value_counts().plot(kind='barh', figsize=(15, 15))
plt.ylabel("Venues", fontsize=20)
plt.xlabel("Number", fontsize=14)
plt.title("Different venues in Kanpur",fontsize=19);


# In[30]:


kan = kanpur_onehot.groupby('neighborhood').mean().reset_index()
kan


# In[31]:


num_top_venues = 5

for hood in kan['neighborhood']:
    print("----"+hood+"----")
    temp = kan[kan['neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# In[36]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['neighborhood'] = kan['neighborhood']

for ind in np.arange(kan.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(kan.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# In[37]:


kclusters = 5

kan_cluster = kan.drop('neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(kan_cluster)

# check cluster labels generated for each row in the dataframe
labels=kmeans.labels_
neighborhoods_venues_sorted['labels']=labels
neighborhoods_venues_sorted


# In[38]:


kan_merged = kanpur

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
kan_merged = kan_merged.join(neighborhoods_venues_sorted.set_index('neighborhood'), on='neighborhood')
kan_merged.dropna(axis=0,inplace=True)
kan_merged["labels"] = kan_merged["labels"].astype(int)
kan_merged.head() # check the last columns!


# In[39]:


# create map
map_clusters = folium.Map(location=[26.4750,80.3083], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(kan_merged['latitude'], kan_merged['longitude'],kan_merged['neighborhood'], kan_merged['labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# ## Analysing Clusters
# 

# In[83]:


#kolkata clusters 
k_merged.loc[k_merged['labels'] == 0, k_merged.columns[[1] + list(range(5,k_merged.shape[1]))]].head()


# In[84]:


k_merged.loc[k_merged['labels'] == 1, k_merged.columns[[1] + list(range(5,k_merged.shape[1]))]]


# In[85]:


k_merged.loc[k_merged['labels'] == 2, k_merged.columns[[1] + list(range(5,k_merged.shape[1]))]]


# In[86]:


#chennai clusters
c_merged.loc[c_merged['labels'] ==0 , c_merged.columns[[1] + list(range(5,c_merged.shape[1]))]].head()


# In[87]:


c_merged.loc[c_merged['labels'] ==1 , c_merged.columns[[1] + list(range(5,c_merged.shape[1]))]].head()


# In[88]:


c_merged.loc[c_merged['labels'] ==2 , c_merged.columns[[1] + list(range(5,c_merged.shape[1]))]].head()


# In[89]:


c_merged.loc[c_merged['labels'] ==3 , c_merged.columns[[1] + list(range(5,c_merged.shape[1]))]]


# In[90]:


c_merged.loc[c_merged['labels'] ==4 , c_merged.columns[[1] + list(range(5,c_merged.shape[1]))]]


# In[91]:


#delhi clusters
d_merged.loc[d_merged['labels'] ==0 , d_merged.columns[[1] + list(range(5,d_merged.shape[1]))]].head()


# In[92]:


d_merged.loc[d_merged['labels'] ==1 , d_merged.columns[[1] + list(range(5,d_merged.shape[1]))]].head()


# In[93]:


d_merged.loc[d_merged['labels'] ==2, d_merged.columns[[1] + list(range(5,d_merged.shape[1]))]]


# In[94]:


d_merged.loc[d_merged['labels'] ==3 , d_merged.columns[[1] + list(range(5,d_merged.shape[1]))]].head()


# In[95]:


d_merged.loc[d_merged['labels'] ==4 , d_merged.columns[[1] + list(range(5,d_merged.shape[1]))]].head()


# In[96]:


#mumbai clusters
m_merged.loc[m_merged['labels'] ==0 , m_merged.columns[[1] + list(range(5,m_merged.shape[1]))]].head()


# In[97]:


m_merged.loc[m_merged['labels'] ==1 , m_merged.columns[[1] + list(range(5,m_merged.shape[1]))]].head()


# In[98]:


m_merged.loc[m_merged['labels'] ==2, m_merged.columns[[1] + list(range(5,m_merged.shape[1]))]].head()


# In[99]:


m_merged.loc[m_merged['labels'] ==3 , m_merged.columns[[1] + list(range(5,m_merged.shape[1]))]].head()


# In[100]:


m_merged.loc[m_merged['labels'] ==4 , m_merged.columns[[1] + list(range(5,m_merged.shape[1]))]].head()


# In[40]:


#kanpur clusters
kan_merged.loc[kan_merged['labels'] ==0 , kan_merged.columns[[1] + list(range(5,kan_merged.shape[1]))]].head()


# In[41]:


kan_merged.loc[kan_merged['labels'] ==1 , kan_merged.columns[[1] + list(range(5,kan_merged.shape[1]))]].head()


# In[42]:


kan_merged.loc[kan_merged['labels'] ==2 , kan_merged.columns[[1] + list(range(5,kan_merged.shape[1]))]].head()


# In[43]:


kan_merged.loc[kan_merged['labels'] ==3 , kan_merged.columns[[1] + list(range(5,kan_merged.shape[1]))]].head()


# In[44]:


kan_merged.loc[kan_merged['labels'] ==4 , kan_merged.columns[[1] + list(range(5,kan_merged.shape[1]))]].head()


# ## Analysing number of distinct venues in each metro

# In[101]:


metro=[k_areas,d_areas,c_areas,m_areas]
metro


# In[102]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(['kolkata','delhi','chennai','mumbai'],metro,color='g')
plt.title("Number of Distinct Venues",fontsize=20)
plt.xlabel("Metros",fontsize=10)
plt.ylabel("Number of Venues",fontsize=10)
plt.show()




