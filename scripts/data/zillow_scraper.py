import requests
from bs4 import BeautifulSoup
import json
import time
import sys
import random

class ZillowScraper:

    # EDIT THIS:
    headers={
            'accept':'',
            'accept-encoding':'',
            'accept-language':'',
            'cache-control':'',
            'cookie':'',
            'sech-ch-ua':'',
            'sech-ch-ua-mobile':'',
            'sec-fetch-dest':'',
            'sec-fetch-mode':'',
            'sec-fetch-site':'',
            'sec-fetch-user':'',
            'upgrade-insecure-requests':'',
            'user-agent':'',
    }

    def fetch(self,url,header,params):
        time.sleep(random.randint(2,5))
        response=requests.get(url, headers=header, params=params)

        print(response, url)

        return response

    def parse(self,response):
        content=BeautifulSoup(response, 'html.parser')
        return content

    def run(self, url, pages_count, dst_file):

        for page in range(1,pages_count+1): # Iterate over pages
            
            # EDIT THIS:
            params={'searchQueryState':'{"pagination":{"currentPage":%s}, "usersSearchTerm":"Cleveland, OH","mapBounds":{"west":-81.8591531176708,"east":-81.750624989777,"south":41.396383768851706,"north":41.52003036520841},"mapZoom":13,"regionSelection":[{"regionId":24115,"regionType":6}],"isMapVisible":true,"filterState":{"pmf":{"value":false},"fore":{"value":false},"ah":{"value":true},"sort":{"value":"globalrelevanceex"},"auc":{"value":false},"nc":{"value":false},"rs":{"value":true},"fsbo":{"value":false},"cmsn":{"value":false},"pf":{"value":false},"fsba":{"value":false}},"isListVisible":true}' %page}


            print("Page: ", page)
            print("Params: ", params)
            
            # Get web page content
            response=self.fetch(url,self.headers,params)
            content=self.parse(response.text)

            deck=content.find('ul', {'class':'photo-cards photo-cards_wow photo-cards_short photo-cards_extra-attribution'})
            
            # Find links to individual real estate listing web pages
            for card in deck.contents:

                script=card.find('script',{'type':'application/ld+json'})

                if script: #Else its an advertisement
                    try:
                        #basic info (available from website listing card)
                        basic_info_dict={}

                        #find date of sale
                        sale_date=card.find('div',{'class':'list-card-variable-text list-card-img-overlay'}).text
                        basic_info_dict['sale_date']=sale_date

                        script_json=json.loads(script.contents[0]) #script doesnt have .text
                        basic_info_dict['latitude']=script_json['geo']['latitude']
                        basic_info_dict['longitude']=script_json['geo']['longitude']
                        basic_info_dict['floorSize']=script_json['floorSize']['value']
                        basic_info_dict['url']=script_json['url']
                        basic_info_dict['price']=card.find('div', {'class': 'list-card-price'}).text
                        # print(json.dumps(basic_info_dict,indent=4, sort_keys=True))

                    
                        # additional info (need to open listing to view)
                        link_response=self.fetch(basic_info_dict['url'],self.headers,params)
                        listing_content=BeautifulSoup(link_response.text, 'html.parser')

                        # facts and features
                        facts_and_features_dict={}

                        for home_fact in listing_content.find('ul',{'class':'ds-home-fact-list'}).contents:
                            # print(home_fact)
                            spans=home_fact.find_all('span')
                            key=spans[0].text
                            value=spans[1].text
                            facts_and_features_dict[key]=value
                        # print(json.dumps(facts_and_features,indent=4, sort_keys=True))


                        # additional features
                        additional_features_dict={}

                        info=listing_content.find('div',{'class':'ds-home-facts-and-features reso-facts-features sheety-facts-features'})
                        info_div=info.div #all description sections (contains facts_and_features, additional_features)
                        additional_features=info_div.find_all('div')[1] #we select additional features
                        for div in additional_features: #iterate over additonal feature sections
                            name=div.h5 #get its name
                            if name: #does not exist for "see more info" div
                                # print(name.text)
                                sections=div.div #get its info
                                subsections=sections.find_all('div')

                                subsections_dict={}

                                for subsection in subsections:
                                    subsection_name=subsection.h6.text
                                    # print('\t',subsection_name)
                                    ul=subsection.ul

                                    items_dict={}

                                    for item in ul.find_all('li'):
                                        try:
                                            if ':' in item.span.text: # items that contain ':' represent categorical variables, those that dont represent bool variables
                                                key,value=item.span.text.split(':')
                                                key=key.strip()
                                                value=value.strip()
                                                items_dict[key]=value
                                            else:
                                                key=item.span.text
                                                value=True
                                                items_dict[key]=value
                                        except:
                                            print('ITEM ERROR:')
                                            print('url',basic_info_dict['url'])
                                            print('description',item.span.text)
                                        # print('\t\t',key,value)

                                    subsections_dict[subsection_name]=items_dict

                                additional_features_dict[name.text]=subsections_dict

                        with open(dst_file,'a') as f:
                            listing_dict={}
                            listing_dict['basic_info']=basic_info_dict
                            listing_dict['facts_and_features']=facts_and_features_dict
                            listing_dict['additional_features']=additional_features_dict
                            f.write(json.dumps(listing_dict)+'\n')
                    except KeyboardInterrupt:
                        print('In order to terminate send SIGINT again')
                        try:
                            time.sleep(1)
                        except KeyboardInterrupt:
                            print('TERMINATED')
                            sys.exit()
                    except:
                        print('Error Fetching Data:', end=' ')
                        try:
                            print(basic_info_dict['url'])
                        except:
                            print('URL not available')


        

if __name__=='__main__':

    # EDIT THIS
    url='https://www.zillow.com/cleveland-oh/'
    pages_count=70
    dst_file='data/raw/zillow_ClevelandOH_SOLD.jsonl'

    scraper=ZillowScraper()
    scraper.run(url, pages_count, dst_file)