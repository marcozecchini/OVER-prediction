import datetime as dt
import requests

#obtain the token
payload = {'grant_type':'password', 'client_id':'mzecchini', 'client_secret':'<client-secret>', 'username':'mzecchini', 'password':'<password>'}
reply = requests.post('https://sso.sparkworks.net/aa/oauth/token', payload)
token = reply.json()['access_token']

#obtain timestamp of yesterday and that one of one year ago
now = dt.datetime.now()
yesterday = (now - dt.timedelta(days=1))
yesterday_timestamp = yesterday.timestamp()
year_ago = (now - dt.timedelta(days=365*2))
year_ago_timestamp = year_ago.timestamp()

ids = []
consumption_file = open('real_consumption.txt', 'w')
temperature_file = open('temperature.txt', 'w')
header = {'Accept' : 'application/hal+json', 'Authorization' : token}
reply = requests.get('https://buildings.gaia-project.eu/gaia-building-knowledge-base/sites', headers=header)
for item in reply.json()['_embedded']['siteContainers']:
    ids += [item['site']['id']]

for id in ids:
    header = {'Accept': 'application/json', 'Authorization': token}
    reply = requests.get('https://analytics.gaia-project.eu/gaia-analytics/statistics/sites/{id}/aggregatedPowerConsumption?from={fro}&to={to}&granularity=HOUR'
                         .format(id=id, fro = int(year_ago_timestamp)*1000, to= int(yesterday_timestamp)*1000), headers=header)
    if(reply.status_code == 200):
        consumption_file.write('{id}\t'.format(id=id))
        for measure in reply.json()['measurements']:
            consumption_file.write(' ')
            consumption_file.write('{timestamp},{reading}'.format(timestamp= measure['timestamp'], reading = measure['reading']))
        consumption_file.write('\n')
consumption_file.close()

for id in ids:
    header = {'Accept': 'application/json', 'Authorization': token}
    reply = requests.get('https://analytics.gaia-project.eu/gaia-analytics/statistics/sites/{id}?from={fro}&to={to}&granularity=HOUR&property={property}'
                         .format(id=id, fro = int(year_ago_timestamp)*1000, to= int(yesterday_timestamp)*1000, property='Temperature'), headers=header)
    if(reply.status_code == 200):
        temperature_file.write('{id}\t'.format(id=id))
        for measure in reply.json()['measurements']:
            temperature_file.write(' ')
            temperature_file.write('{timestamp},{reading}'.format(timestamp= measure['timestamp'], reading = measure['reading']))
        temperature_file.write('\n')
temperature_file.close()