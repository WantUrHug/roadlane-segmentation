# roadlane-segmentation
用语义分割的方式来做车道线检测

首先是数据处理，我是用 Anaconda 配置 labelme 逐张进行批注的，每张图片会得到一个json文件，包含了一些必要的信息，将他们保存在同个文件夹中，如下</br>
D:</br>
├─00001.json</br>
├─00002.json</br>
├─00003.json</br>
把 build.bat 放在同个路径下，先 activate labelme，然后就执行这个批处理文件即可</br>
D:</br>
├─00001_json (subdir)</br>
├─00001.json</br>
├─00002_json (subdir)</br>
├─00002.json</br>
├─00003_json (subdir)</br>
├─00003.json</br>
