import schedule

def Calc_CRC_array(command_): #CRC配列を計算するメソッド
    # コマンドのバイト配列を生成する
    command = bytearray.fromhex(command_)

    # CRC の初期値を 0xFFFF にする
    crc = 0xFFFF
    # CRC のビット配列を出力する
    # print(format(crc, '>16b'))

    # p10 ここからp10_endまでをコマンドの全てのバイトで実行するようにループ.(インデントを下げる)
    # パケットの最後まで処理していなければ, CRCレジスタとパケットの次の8bitsのXORを計算してCRCレジスタに戻し, 3の手順から繰り返す.
    for i in range(len(command)):

    # p6 CRCと初めの8bit(1byte)のXOR計算する
        crc = crc^command[i] # p10を実装する際はcommand[i]
    # print(format(crc, '>16b'))


    # p9 ここからp9_endまでを8回ループ.(インデントを下げる)
    # 8bits分ビットシフトするまで3,4の手順を繰り返す.
        for j in range(8):

    # p8 
            flag = crc & 1

    # p7 MSBを"0"で埋めながら, CRCレジスタを1bit右シフトする.
    # >> or << でビットシフト.
            crc = crc >> 1
    # print(format(crc, '>16b'))

    # p8
    # LSBからシフトされたbitが"0"ならば, 3の手順を繰り返す.
    # "1"ならばCRCレジスタと0xA001でXOR計算し, 結果をCRCレジスタに戻す.
            if flag == 1:
                crc = crc ^ 0xA001
    # print(format(crc, '>16b'))

    # p9_end(ループはここまで)
    # print(format(crc, '>16b'))

    # p10_end(ループはここまで)
    # print(format(crc, '>16b'))

    # p10
    # print(format(crc, 'x'))

    # p11 CRC配列の順番を逆転させて, コマンド配列に追加.
    command += bytearray([crc & 0x00FF, crc >> 8])
    return command.hex()


def getSensor(command_):
    import serial
    import time
    import requests
    import json
    import paho.mqtt.client as mqtt
    
    # センサを準備する
    sensor = serial.Serial("/dev/ttyUSB0", 115200, serial.EIGHTBITS, serial.PARITY_NONE)
    # センサの値を読み取る
    sensor.write(bytearray.fromhex(Calc_CRC_array(command_)))
    time.sleep(0.2)
    data = sensor.read(sensor.inWaiting())
    # print(data.hex())
    
    # センサから返却された値を解釈する
    temperature = (data[9] * 256 + data[8]) * 0.01
    humidity = (data[11] * 256 + data[10]) * 0.01
    light = data[13] * 256 + data[12]
    pressure = (data[17] * (256**3)  + data[16] * (256**2)  + data[15] * 256 + data[14]) * 0.001
    noise = (data[19] * 256 + data[18]) * 0.01
    eCO2 = data[23] * 256 + data[22]
    PGA = (data[32] * 256 + data[31]) * 0.1
    print("Temperature = {0} degC".format(temperature))
    print("Humidity = {0} %RH".format(humidity))
    print("Light = {0} lx".format(light))
    print("Pressure = {0} hPa".format(pressure))
    print("Noise = {0} db".format(noise))
    print("eCO2= {0} ppm".format(eCO2))
    print("PGA = {0} gal".format(PGA))
    print()



    # APIにセンサ値を送信する
    body = {"id":0, "temperature":temperature, "humidity":humidity, "noise":noise, "eCO2":eCO2}
    client = mqtt.Client()
    client.connect("10.145.146.49", 1883, 60)
    client.loop_start()
    client.publish("sensor/value", json.dumps(body))

command = "52420500012150"
schedule.every(6).seconds.do(getSensor, command_ = command)

if __name__ == "__main__":    
    import time
    while True : 
        schedule.run_pending()
        time.sleep(1)


