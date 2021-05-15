# 0 9 * * * /home/ec2-user/update.sh
wget -O data/nhk_news_covid19_prefectures_daily_data.csv.raw url https://www3.nhk.or.jp/n-data/opendata/coronavirus/nhk_news_covid19_prefectures_daily_data.csv
./trim_csvdata.sh data/nhk_news_covid19_prefectures_daily_data.csv.raw > data/nhk_news_covid19_prefectures_daily_data.csv
python3 generate_csv.py
python3 generate_csv_tokyo.py
