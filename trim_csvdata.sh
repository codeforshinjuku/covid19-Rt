#!/usr/bin/php7.4
<?php
$csv_file = $_SERVER['argv'][1] ?? '';
if (file_exists($csv_file) && !is_dir($csv_file)){
    $limit_date = strtotime(date('Ymd')) - 86400 * 365;
    if ($fp = fopen($csv_file, 'r')){
        while ($line = fgets($fp, 1024)){
            $cell = explode(",", $line);
            if (preg_match('@^\d{4}/\d{1,2}/\d{1,2}$@', $cell[0])){
                if (strtotime($cell[0]) < $limit_date){
                    continue;
                }
            }
            echo $line;
        }
    }
}

