import flask
import mysql.connector

app = flask.Flask(__name__)

def Calc_discomfort(temperature, humidity):
    # 不快指数の計算
    discomfort = 0.81*temperature + 0.01*humidity * (0.99*temperature - 14.3) + 46.3
    
    if discomfort >= 85:
        return 1
    elif (discomfort < 55) or ((discomfort >= 80) and (discomfort < 85)): 
        return 2
    elif ((discomfort >= 55) and (discomfort < 60)) or ((discomfort >= 55) and (discomfort < 60)):
        return 3
    elif ((discomfort >= 60) and (discomfort < 65)) or ((discomfort >= 70) and (discomfort < 75)):
        return 4
    elif (discomfort >= 65) and (discomfort < 70):
        return 5
    else:
        return 0

def Calc_noise(noise):
    if noise >= 80:
        return 1
    elif (noise >= 60) and (noise < 80):
        return 2
    elif (noise >= 40) and (noise < 60):
        return 3
    elif (noise >= 20) and (noise < 40):
        return 4
    elif noise < 20:
        return 5
    else:
        return 0

def Calc_eCO2(eCO2):
    if eCO2 >= 2500:
        return 1
    elif (eCO2 >= 1500) and (eCO2 < 2500):
        return 2
    elif (eCO2 >= 1000) and (eCO2 < 1500):
        return 4
    elif (eCO2 < 1000):
        return 5
    else:
        return 0



@app.route("/")
def serch():
    # それっぽく見せるためにname配列とscores配列を用意
    name=["Room 115", "Room 621", "Room 111", "WorkLab"]
    scores = []
    # データベースに接続する
    context = mysql.connector.connect(user="app", password="Abcde123")
    cursor = context.cursor()       
    # データベースから値を取得するSQLを実行する
    # place0
    sql_string = f"""
        SELECT * FROM iot.value WHERE id = 0 ORDER BY time DESC LIMIT 4;
    """
    cursor.execute(sql_string)
    items0 = cursor.fetchall()
    
    sql_string = f"""
        SELECT * FROM iot.value WHERE id = 0 AND type = 'temperature' ORDER BY time DESC LIMIT 1;
    """
    cursor.execute(sql_string)
    temperature = cursor.fetchall()
    
    sql_string = f"""
        SELECT * FROM iot.value WHERE id = 0 AND type = 'humidity' ORDER BY time DESC LIMIT 1;
    """
    cursor.execute(sql_string)
    humidity = cursor.fetchall()
    
    sql_string = f"""
        SELECT * FROM iot.value WHERE id = 0 AND type = 'noise' ORDER BY time DESC LIMIT 1;
    """
    cursor.execute(sql_string)
    noise = cursor.fetchall()
    
    sql_string = f"""
        SELECT * FROM iot.value WHERE id = 0 AND type = 'eCO2' ORDER BY time DESC LIMIT 1;
    """
    cursor.execute(sql_string)
    eCO2 = cursor.fetchall()
    
    # 独自評価を要素に追加
    score = Calc_discomfort(temperature[0][2], humidity[0][2]) + Calc_noise(noise[0][2]) + Calc_eCO2(eCO2[0][2])
    scores.append(score)
    
    # place1
    sql_string = f"""
        SELECT * FROM iot.value WHERE id = 1 ORDER BY time DESC LIMIT 4;
    """
    cursor.execute(sql_string)
    items1 = cursor.fetchall()
    
    sql_string = f"""
        SELECT * FROM iot.value WHERE id = 1 AND type = 'temperature' ORDER BY time DESC LIMIT 1;
    """
    cursor.execute(sql_string)
    temperature = cursor.fetchall()
    
    sql_string = f"""
        SELECT * FROM iot.value WHERE id = 1 AND type = 'humidity' ORDER BY time DESC LIMIT 1;
    """
    cursor.execute(sql_string)
    humidity = cursor.fetchall()
    
    sql_string = f"""
        SELECT * FROM iot.value WHERE id = 1 AND type = 'noise' ORDER BY time DESC LIMIT 1;
    """
    cursor.execute(sql_string)
    noise = cursor.fetchall()
    
    sql_string = f"""
        SELECT * FROM iot.value WHERE id = 1 AND type = 'eCO2' ORDER BY time DESC LIMIT 1;
    """
    cursor.execute(sql_string)
    eCO2 = cursor.fetchall()
    
    score = Calc_discomfort(temperature[0][2], humidity[0][2]) + Calc_noise(noise[0][2]) + Calc_eCO2(eCO2[0][2])
    scores.append(score)
    
    # place2
    sql_string = f"""
        SELECT * FROM iot.value WHERE id = 2 ORDER BY time DESC LIMIT 4;
    """
    cursor.execute(sql_string)
    items2 = cursor.fetchall()
    
    sql_string = f"""
        SELECT * FROM iot.value WHERE id = 2 AND type = 'temperature' ORDER BY time DESC LIMIT 1;
    """
    cursor.execute(sql_string)
    temperature = cursor.fetchall()
    
    sql_string = f"""
        SELECT * FROM iot.value WHERE id = 2 AND type = 'humidity' ORDER BY time DESC LIMIT 1;
    """
    cursor.execute(sql_string)
    humidity = cursor.fetchall()
    
    sql_string = f"""
        SELECT * FROM iot.value WHERE id = 2 AND type = 'noise' ORDER BY time DESC LIMIT 1;
    """
    cursor.execute(sql_string)
    noise = cursor.fetchall()
    
    sql_string = f"""
        SELECT * FROM iot.value WHERE id = 2 AND type = 'eCO2' ORDER BY time DESC LIMIT 1;
    """
    cursor.execute(sql_string)
    eCO2 = cursor.fetchall()
    
    score = Calc_discomfort(temperature[0][2], humidity[0][2]) + Calc_noise(noise[0][2]) + Calc_eCO2(eCO2[0][2])
    scores.append(score)
    
    # place3
    sql_string = f"""
        SELECT * FROM iot.value WHERE id = 3 ORDER BY time DESC LIMIT 4;
    """
    cursor.execute(sql_string)
    items3 = cursor.fetchall()
    
    sql_string = f"""
        SELECT * FROM iot.value WHERE id = 3 AND type = 'temperature' ORDER BY time DESC LIMIT 1;
    """
    cursor.execute(sql_string)
    temperature = cursor.fetchall()
    
    sql_string = f"""
        SELECT * FROM iot.value WHERE id = 3 AND type = 'humidity' ORDER BY time DESC LIMIT 1;
    """
    cursor.execute(sql_string)
    humidity = cursor.fetchall()
    
    sql_string = f"""
        SELECT * FROM iot.value WHERE id = 3 AND type = 'noise' ORDER BY time DESC LIMIT 1;
    """
    cursor.execute(sql_string)
    noise = cursor.fetchall()
    
    sql_string = f"""
        SELECT * FROM iot.value WHERE id = 3 AND type = 'eCO2' ORDER BY time DESC LIMIT 1;
    """
    cursor.execute(sql_string)
    eCO2 = cursor.fetchall()
    
    score = Calc_discomfort(temperature[0][2], humidity[0][2]) + Calc_noise(noise[0][2]) + Calc_eCO2(eCO2[0][2])
    scores.append(score)
    

    # 後始末
    cursor.close()
    context.close()
    

    return flask.render_template(
        "top.html", 
        name=name, 
        items0=items0, 
        items1=items1, 
        items2=items2, 
        items3=items3, 
        score = scores
    )

# サーバーの起動
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
