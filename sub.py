import paho.mqtt.client as mqtt
import json
import mysql.connector

#MQTTブローカーに接続された場合に実装されるメソッド
def on_connect(client, userdata, flag, rc):
        print("Connected")

        #トピックを指定して, MQTTメッセージをサブスクライブする
        client.subscribe("sensor/value")

#MQTTメッセージを受け取った際に実行されるメソッド
def on_message(client, userdata, msg):
        #print(msg.payload)
        data = json.loads(msg.payload)
        id = data["id"]
        temperature = data["temperature"]
        humidity = data["humidity"]
        noise = data["noise"]
        eCO2 = data["eCO2"]
        
        # データベースに接続する
        context = mysql.connector.connect(user="", password="")
        cursor = context.cursor()       
        
        # JSONから取得した値を使用してSQLを実行する
        cursor.execute("INSERT INTO iot.value VALUES (%s, 'temperature', %s, now())", (id, temperature))
        cursor.execute("INSERT INTO iot.value VALUES (%s, 'humidity', %s, now())", (id, humidity))
        cursor.execute("INSERT INTO iot.value VALUES (%s, 'noise', %s, now())", (id, noise))
        cursor.execute("INSERT INTO iot.value VALUES (%s, 'eCO2', %s, now())", (id, eCO2))
        context.commit()
        
        # 後始末
        cursor.close()
        context.close()
        print(f"Message Recieved (id, temperature, humidity, noise, eCO2) = ({id}, {temperature}degC, {humidity}%RH, {noise}dB, {eCO2}ppm")


#MQTT ブローカーに接続する
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect("10.145.146.49", 1883, 60)

#MQTTメッセージがパブリッシュされるのを待つ(永久)
client.loop_forever()