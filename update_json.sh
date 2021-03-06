#!/usr/bin/php
<?php
chdir(__DIR__);
$citylist =  [
    '131016' => '千代田区', '131024' => '中央区', '131032' => '港区', '131041' => '新宿区', '131059' => '文京区', '131067' => '台東区', '131075' => '墨田区',
    '131083' => '江東区', '131091' => '品川区', '131105' => '目黒区', '131113' => '大田区', '131121' => '世田谷区', '131130' => '渋谷区', '131148' => '中野区',
    '131156' => '杉並区', '131164' => '豊島区', '131172' => '北区', '131181' => '荒川区', '131199' => '板橋区', '131202' => '練馬区', '131211' => '足立区',
    '131229' => '葛飾区', '131237' => '江戸川区', '132012' => '八王子市', '132021' => '立川市', '132039' => '武蔵野市', '132047' => '三鷹市', '132055' => '青梅市',
    '132063' => '府中市', '132071' => '昭島市', '132080' => '調布市', '132098' => '町田市', '132101' => '小金井市', '132110' => '小平市', '132128' => '日野市',
    '132136' => '東村山市', '132144' => '国分寺市', '132152' => '国立市', '132187' => '福生市', '132195' => '狛江市', '132209' => '東大和市', '132217' => '清瀬市',
    '132225' => '東久留米市', '132233' => '武蔵村山市', '132241' => '多摩市', '132250' => '稲城市', '132276' => '羽村市', '132284' => 'あきる野市', '132292' => '西東京市',
    '133035' => '瑞穂町', '133051' => '日の出町', '133078' => '檜原村', '133086' => '奥多摩町', '133612' => '大島町', '133621' => '利島村', '133639' => '新島村',
    '133647' => '神津島村', '133817' => '三宅村', '133825' => '御蔵島村', '134015' => '八丈町', '134023' => '青ヶ島村', '134210' => '小笠原村'];

$preflist = [
    1 => '北海道', '青森県', '岩手県', '宮城県', '秋田県', '山形県', '福島県', '茨城県', '栃木県', '群馬県', '埼玉県', '千葉県',
    '東京都', '神奈川県', '新潟県', '富山県', '石川県', '福井県', '山梨県', '長野県', '岐阜県', '静岡県', '愛知県', '三重県',
    '滋賀県', '京都府', '大阪府', '兵庫県', '奈良県', '和歌山県', '鳥取県', '島根県', '岡山県', '広島県', '山口県', '徳島県',
    '香川県', '愛媛県', '高知県', '福岡県', '佐賀県', '長崎県', '熊本県', '大分県', '宮崎県', '鹿児島県', '沖縄県'];
$data_files = [
    'tokyo' => [
        'list' => $citylist,
        'csv'  => 'dist/rt_tokyo.csv',
        'json' => 'dist/rt_tokyo.json',
        ],
    'japan' => [
        'list' => $preflist,
        'csv' => 'dist/rt_japan.csv',
        'json' => 'dist/rt_japan.json',
        ]
    ];
foreach ($data_files as $prop) {
$json = [
    'latest' => [],
    'data'   => [],
    ];
$code = array_flip($prop['list']);
$data = [];
$header = [];
if ($fp = fopen($prop['csv'], 'r')){
    while ($csv = fgetcsv($fp)){
        if (!$header){
            $header = [1 =>$csv[1], $csv[2], $csv[3], $csv[4], $csv[5], $csv[6]];
        }
        else {
            $city = $code[$csv[0]];
            if (!isset($data[$city])){
                $data[$city] = [];
            }
            $date_data = [];
            foreach ($header as $i=>$key){
                $date_data[$key] = $csv[$i];
            }
            $data[$city][]= $date_data;
        }
    }
}

foreach ($data as $city => $d){
    $tmp = end($d);
    $tmp['city'] = $city;
    $json['latest'][] = $tmp;
}
usort($json['latest'], function($a, $b){
    return $a['ML'] >= $b['ML'];
});
foreach (array_keys($prop['list']) as $code){
    if (isset($data[$code])){
        $json['data'][$code] = $data[$code];
    }
}
file_put_contents($prop['json'], json_encode($json));
}
