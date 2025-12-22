需要配置的是图片文件的相对根目录，它会自动解析这个目录，并且使用每个最小子目录的第一张图片，也就是每个轨迹的首张图片。
我有写写入文件的代码，不过这部分还没验证。
最后存成jsonl的格式差不多是这样的：
{
"image":<image path>,
"task":<task>,
"reasoning":<reasoning>,
"plan":<plan>
}
