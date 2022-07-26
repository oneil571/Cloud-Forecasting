


#RCWO

import eumdac
import datetime 
import shutil

#please don't make me regret putting my keys here:
consumer_key = 'OZaZCvl5v6xRgfQxoBUdbJjuu5Aa'
consumer_secret = '9peRxifg9gsoP2f13sBvWnvrGmQa'

credentials = (consumer_key, consumer_secret)

token = eumdac.AccessToken(credentials)

print(f"This token '{token}' expires {token.expiration}")

datastore = eumdac.DataStore(token)
datastore.collections


#select cloud masks:
name = 'EO:EUM:DAT:MSG:CLM'

selected_collection = datastore.get_collection(name)

# Add vertices for polygon, wrapping back to the start point.
geometry = [[-1.0, -1.0],[4.0, -4.0],[8.0, -2.0],[9.0, 2.0],[6.0, 4.0],[1.0, 5.0],[-1.0, -1.0]]

# Set sensing start and end time
start = datetime.datetime(2022, 4, 4, 0, 0)
end = datetime.datetime(2022, 5, 4, 0, 0)

# Retrieve datasets that match our filter
products = selected_collection.search(
    geo='POLYGON(({}))'.format(','.join(["{} {}".format(*coord) for coord in geometry])),
    dtstart=start, 
    dtend=end)
  
print(f'Found Datasets: {len(products)} datasets for the given time range')

'''
for product in products:
    print(str(product))
'''
  
i = 0
tot = len(products)
for product in products:
    i+=1
    with product.open() as fsrc, \
            open(fsrc.name, mode='wb') as fdst:
        shutil.copyfileobj(fsrc, fdst)
        print(i/tot,f'Download of product {product} finished.')
    
print('All downloads are finished.')
