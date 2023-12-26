import asyncio, requests
import aiohttp
import urllib.request, urllib.error
import numpy as np
import time
import urllib.request
t1 = time.time()
filename = '123.txt'
xml_papers = []
doc=[]
# data = np.loadtxt(filename,dtype=str)
# xml_papers = data.tolist()
baseurl = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/"
addurl= "/unicode"
# for k, article in enumerate(xml_papers):
#     xml_papers[k] = baseurl + str(article) + addurl



async def job(session,url):
    filename= url.split("/")[-2]+".xml"
    ftp=await session.get(url)
    ftpcode=await ftp.read()
    with open("test/"+str(filename),'wb') as f:
        f.write(ftpcode)
    return str(url)

    # urllib.request.urlretrieve(url, filename)
connector = aiohttp.TCPConnector(force_close=True)  # 禁用 HTTP keep-alive
async def getXML(loop,URL):
    async with aiohttp.ClientSession(connector=connector)as session:
        tasks = [loop.create_task(job(session, URL[_])) for _ in range(len(URL))]
        # 建立所有任务
        finished, unfinished = await asyncio.wait(tasks)# 触发await，等待任务完成
        all_results = [r.result() for r in finished]# 获取所有结果
        print("ALL RESULT:"+str(all_results))



t2 = time.time()
print(int(t2-t1))
def getFTP(xml_papers):
    for k, article in enumerate(xml_papers):
        xml_papers[k] = baseurl + str(article) + addurl
    loop = asyncio.get_event_loop()
    loop.run_until_complete(getXML(loop, xml_papers))
    loop.close()


