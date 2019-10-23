# roadlane-segmentation
用语义分割的方式来做车道线检测

首先是数据处理，我是用 Anaconda 配置 labelme 逐张进行批注的，每张图片会得到一个json文件，包含了一些必要的信息，将他们保存在同个文件夹中，如下
D:.
├─00001.json
├─00002.json
├─00003.json
把 build.bat 放在同个路径下，先 activate labelme，然后就执行这个批处理文件即可
D:.
├─00001_json (subdir)
├─00001.json
├─00002_json (subdir)
├─00002.json
├─00003_json (subdir)
├─00003.json
