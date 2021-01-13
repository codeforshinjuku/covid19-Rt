# 0 9 * * * /home/ec2-user/update.sh
python3 generate_csv_tokyo.py
wget -O data/nhk_news_covid19_prefectures_daily_data.csv url https://www3.nhk.or.jp/n-data/opendata/coronavirus/nhk_news_covid19_prefectures_daily_data.csv
python3 generate_csv.py
