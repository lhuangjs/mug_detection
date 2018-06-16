import os
import shutil
def splitDataset(sourceDir):
    """
    将源数据分割为train set(剩余所有)、validation set(10)、test set(10)
    sourceDir: 源数据目录路径
    """
    import random

    # 在当前dataset目录下创建目录train、val、test存储train set、validation set、test set，若存在则删除
    datasetDir = ["./dataset/train", "./dataset/val", "./dataset/test"]
    for dir in datasetDir:
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.mkdir(dir)
        print("%s创建成功" % dir)
    
    # 分割数据集
    dataset = os.listdir(sourceDir)
    random.seed(1)
    random.shuffle(dataset)
    
    testSet = dataset[:10]
    copyFileList(sourceDir, testSet, datasetDir[2])

    valSet = dataset[10 : 20]
    copyFileList(sourceDir, valSet, datasetDir[1])

    trainSet = dataset[20 : ]
    copyFileList(sourceDir, trainSet, datasetDir[0])

def copyFileList(sourceDir, fileList, targetDir):
    """
    复制文件列表(fileList)中来自源目录(sourceDir)的文件到目标目录(targetDir)
    """
    for f in fileList:
        shutil.copyfile(os.path.join(sourceDir, f), os.path.join(targetDir, f))

    print("=====%d个文件从%s复制到%s中=====" % (len(fileList), sourceDir, targetDir))

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='数据预处理')
    parser.add_argument("--source", required=True,
                        help="保存源数据的目录")
    args = parser.parse_args()
    splitDataset(args.source)
