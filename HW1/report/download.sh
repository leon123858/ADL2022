zipId = 
# 獲取確認下載的狀態所夾帶的 cookie
wget --save-cookies cookies.txt "https://docs.google.com/uc?export=download&id=19R3t4-KSv119OAKAAeOK3myCv2r78buJ" -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt
# 利用 cookie 順利下載
wget --load-cookies cookies.txt -O ADL_HW1_data.zip \
     'https://docs.google.com/uc?export=download&id=19R3t4-KSv119OAKAAeOK3myCv2r78buJ&confirm='$(<confirm.txt)
# 原地解壓縮, 下載完成前處理結果以及模型
unzip ADL_HW1_data.zip