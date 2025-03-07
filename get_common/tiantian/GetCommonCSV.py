import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import json
import re


# 收集指定50个基金的用户评论数据，数据限定范围
# 1.作者不为:基金资讯
# 2.最新更新时间:24Q4的帖子
# 需要包含的内容
# 1.帖子发表日期
# 2.帖子题目和文字内容
# 3.帖子下文字评论，
# 4.评论时间
# 注意:同一个帖子下面的所有评论，都算在这个帖子的评论中采集

class FundCrawler:
    def __init__(self, headers, home_url):
        self.headers = headers
        self.home_url = home_url
        self.skip_user_nickname = "基金资讯"
        self.filepath = "C:/Users/HP/Desktop/commons_data.csv"

    def get_comment_basic_info(self, df):
        """ 获取评论的基本信息，包括评论页面的URL """
        try:
            basic_url = "http://guba.eastmoney.com/list,of"
            fund_code = df['tradingcode'].iloc[0]  # 假设只抓取基金001765的评论
            print(f'Processing fund {fund_code}...')

            init_url = f"{basic_url}{fund_code}.html"
            r = requests.get(init_url, headers=self.headers)
            soup = BeautifulSoup(r.text, 'lxml')
            data_pager = soup.find("div", class_="pager")
            
            # 确保 'pager' 存在
            if not data_pager:
                print(f"Error: No pagination info found for fund {fund_code}")
                return df
            
            data_pager = data_pager.span.get('data-pager')
            article_sum = int(data_pager.split('|')[1])
            page_split = int(data_pager.split('|')[2])
            page_sum = int(article_sum / page_split)

            basic_comment_url = [f"{basic_url}{fund_code}_{num}.html" for num in range(1, page_sum + 1)]
            
            # 确保 'basic_comment_url' 被正确赋值
            df['page_sum'] = page_sum
            df['basic_comment_url'] = [basic_comment_url] * len(df)  # 确保每个基金都有基本评论URL列表
            print(f"Successfully fetched {len(basic_comment_url)} comment URLs for fund {fund_code}.")
            return df
        except Exception as e:
            print(f"Error fetching basic comment info: {e}")
            return df

    def get_comment_detail_url(self, fund_dataframe):
        """ 获取每个评论的详细URL """
        fund_comment_dataframe = fund_dataframe.copy()
        detail_comment_url = []
        try:
            for fund_code in fund_dataframe['tradingcode']:
                # 确保从 'basic_comment_url' 获取到正确的值
                basic_comment_url = fund_dataframe.loc[fund_dataframe['tradingcode'] == fund_code, 'basic_comment_url'].values[0]
                
                if not basic_comment_url:
                    print(f"No basic comment URL for fund {fund_code}, skipping.")
                    detail_comment_url.append([])  # 如果没有基本评论URL，添加空列表
                    continue
                
                tmp_urls = []
                for comment_url in basic_comment_url:
                    r = requests.get(comment_url, headers=self.headers)
                    soup = BeautifulSoup(r.text, 'lxml')
                    comment_all = soup.find_all('span', class_='l3')
                    comment_all = [self.home_url + t for t in [x.a.get('href') for x in comment_all if x.a] if '/' in t]
                    tmp_urls.extend(comment_all)
                detail_comment_url.append(tmp_urls)
            
            # 确保 'detail_comment_url' 被正确赋值
            fund_comment_dataframe['detail_comment_url'] = detail_comment_url
            print(f'Successfully fetched {len(detail_comment_url)} detail comment URLs.')
            return fund_comment_dataframe
        except Exception as e:
            print(f"Error fetching comment detail URLs: {e}")
            return fund_comment_dataframe

    def get_common_from_pages(self,url_page, out_dataframe, fund_code, title, user_nickname):
        r = requests.get(url_page, headers=self.headers)
        reply_list = r.text.split("var reply_list=")[1]
        full_message = reply_list.split("</script>")[0].split("var")[0]
        json_string = full_message.strip().rstrip(";")
        common_message_json = json.loads(json_string)
        point_res = common_message_json["re"]
        for reply_common in point_res:
            reply_publish_time = reply_common["reply_publish_time"]
            reply_text = reply_common["reply_text"]
            out_dataframe.loc[len(out_dataframe)] = [fund_code, title, user_nickname,
                                                     reply_text, reply_publish_time]

    def get_comment(self, fund_comment_dataframe):
        """ 获取基金的评论标题，最多抓取100条评论 """
        out_dataframe = pd.DataFrame(columns=['fund_code','title','user_nickname','common_message','reply_time'])
        try:
            for fund_code in fund_comment_dataframe['tradingcode']:
                # 确保从 'detail_comment_url' 获取到正确的值
                urls = fund_comment_dataframe.loc[fund_comment_dataframe['tradingcode'] == fund_code, 'detail_comment_url'].values[0]
                print(f"Processing fund {fund_code}, found {len(urls)} comment URLs.")
                if urls:
                    # 只抓取前100条评论
                    urls_to_process = urls[:10]
                    for url in urls_to_process:
                        try:
                            r = requests.get(url, headers=self.headers)
                            soup = BeautifulSoup(r.text, 'lxml')
                            data_zwcontt = soup.find('div', id='zwcontt')
                            data_div =data_zwcontt.find('div', class_='data')
                            if data_div and data_div.has_attr('data-json'):
                                data_json = json.loads(data_div['data-json'])  # 解析 JSON
                                user_nickname = data_json.get('user_nickname', '未知')  # 获取 user_nickname
                                if self.skip_user_nickname in user_nickname:
                                    print(f"因发布用户为{self.skip_user_nickname},所以跳过本条咨询")
                                    continue
                                else:
                                    print("用户昵称:", user_nickname)
                                    # 假设评论标题在 <title="title"> 标签内
                                    title = soup.title.get_text() if soup.title else 'null title'
                                    print(title)

                                    reply_list = r.text.split("var reply_list=")[1]
                                    full_message = reply_list.split("</script>")[0].split("var")[0]
                                    json_string = full_message.strip().rstrip(";")
                                    common_message_json = json.loads(json_string)
                                    common_count = common_message_json["count"]
                                    print(f"评论总条数{common_count}")
                                    # point_re 热门评论;re 全部评论
                                    point_res = common_message_json["re"]

                                    #默认为一页的情况
                                    for reply_common in point_res:
                                        reply_publish_time = reply_common["reply_publish_time"]
                                        reply_text = reply_common["reply_text"]
                                        out_dataframe.loc[len(out_dataframe)] = [fund_code, title, user_nickname,
                                                                                 reply_text, reply_publish_time]
                                    #当存在大于一页的情况，查询后续页面
                                    if common_count > 30:
                                        pages = common_count // 30
                                        if common_count % 30 > 0:
                                            pages += 1
                                        for page in range(2, pages + 1):
                                            url_page = str(url).replace(".html",f"_{page}.html")
                                            self.get_common_from_pages(url_page, out_dataframe, fund_code, title,
                                                                       user_nickname)
                                            time.sleep(3)

                            # 打印进度
                            print(f"Successfully fetched comment from {url}")
                            time.sleep(5)  # 给每个请求加个延时，防止过于频繁
                        except Exception as e:
                            print(f"Error fetching comment from {url}: {e}")

            # 写入 CSV 文件
            out_dataframe.to_csv(self.filepath, index=False, encoding="utf-8")
            print("数据已成功写入 data.csv 文件！")
            print('Successfully fetched all comment titles and saved to fund_comments.csv.')
            return out_dataframe
        except Exception as e:
            print(f"Error fetching comments: {e}")
            return out_dataframe


# 示例初始化
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

home_url = "http://guba.eastmoney.com"

# 示例基金数据框架
fund_data = pd.DataFrame({
    'tradingcode': ['001765'],
    'fundname': ['Fund Name'],
    'fundbrief': ['Fund Brief']
})

# 创建爬虫对象并抓取数据
crawler = FundCrawler(headers, home_url)
fund_data = crawler.get_comment_basic_info(fund_data)
fund_data = crawler.get_comment_detail_url(fund_data)
comment_data = crawler.get_comment(fund_data)
