import requests
import re
import pandas as pd

urls = []
for i in range(1000, 50001, 1000):
    r = requests.get('http://stuffgate.com/stuff/website/top-' + str(i) + '-sites')
    https = re.findall(r'<td>([0-9]+)</td>\r\n\t<td><a href="https://(.*)"', r.text)
    http = re.findall(r"<td>([0-9]+)</td>\r\n\t<td><a href=\'http://(.*?)\'", r.text)
    urls.extend(https)
    urls.extend(http)
    print(f'正在抓取{i-1000}-{i}')

urls.sort(key=lambda x: int(x[0]))

frame = pd.DataFrame(urls, columns=['rank', 'domain'])
frame.to_csv("AlexaTop50000.csv")