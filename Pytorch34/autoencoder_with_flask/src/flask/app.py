# coding:utf-8
 
from flask import Flask, render_template, request, redirect,make_response,jsonify
import os
from PIL import Image
import time
from process import process_fruit_images_using_autoencoder
from datetime import timedelta
import matplotlib.pyplot as plt
 
 
#设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
 
app = Flask(__name__)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)
 
 

@app.route('/', methods=['POST', 'GET'])  # 添加路由
def upload():
	#上传图片
    if request.method == 'POST':
        f = request.files['file']
 
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})
 		
 		#得到图片文件名
        user_input = request.form.get("name")
 		
 		#保存上传的文件
        basepath = os.path.dirname(__file__)  # 当前文件所在路径
 
        upload_path = os.path.join(basepath, 'static/images', f.filename)  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        f.save(upload_path)

        # 使用Opencv转换一下图片格式和名称
        if os.path.exists("./static/images/test.png"):
            os.remove("./static/images/test.png")
        os.rename(upload_path, os.path.join(basepath, 'static/images', 'test.png'))
        process_fruit_images_using_autoencoder()
        image = Image.open("./static/images/test2.png")

        return render_template('html2.html',userinput=user_input,val1=time.time(), image=image)

    return render_template('html1.html')

if __name__ == '__main__':
    # app.debug = True
    app.run(host='127.0.0.1', port=8080, debug=True)
