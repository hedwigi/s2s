s2s en-fr tutorial
- bug1: target length after padding, resulting in wrong loss/accuracy
https://github.com/deep-diver/EN-FR-MLT-tensorflow/blob/master/dlnd_language_translationv2.ipynb

tf timeline
https://towardsdatascience.com/howto-profile-tensorflow-1a49fb18073d

official nmt tutorial
https://github.com/tensorflow/nmt#other-details-for-better-nmt-models

stacked bilstm
https://blog.csdn.net/u012436149/article/details/71080601

data clustering
### 过滤掉clusters.center中大量0.0的feat
egrep -no '[^ ]+:0\.0[1-9]' clusters.centers > clusters.centers.not0

### 从dialog id查找原始dialog
egrep -n '^\s*$' multi_1_4.4_100w.data |sed -n 1804,1805p
```
9633:
9637:
```
sed -n 9622,9633p multi_1_4.4_100w.data
```
割了
是淋巴。。

翻到学校之后除左训觉之外没其他野好做……[抓狂]
上课呢?
今日没课[泪]…
哦··这样子···
你今日有课??
没啊··我听日先开始上课···
唉……我都系……
没办法啦···

```