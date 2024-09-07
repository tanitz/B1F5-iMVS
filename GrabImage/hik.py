import sys
import numpy as np
import cv2
from PIL import Image

# sys.path.append(r".\MvImport")
from MvCameraControl_class import *
from CamOperation_class import CameraOperation

def Enum_device(tlayerType, deviceList):
    """
    ch:枚举设备 | en:Enum device
    nTLayerType [IN] 枚举传输层 ，pstDevList [OUT] 设备列表
    """
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0:
        print("enum devices fail! ret[0x%x]" % ret)
        sys.exit()

    if deviceList.nDeviceNum == 0:
        print("find no device!")
        sys.exit()

    print("Find %d devices!" % deviceList.nDeviceNum)

    for i in range(0, deviceList.nDeviceNum):
        mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
            print("\ngige device: [%d]" % i)
            # 输出设备名字
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                strModeName = strModeName + chr(per)
            print("device model name: %s" % strModeName)
            # 输出设备ID
            nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
            nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
            nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
            nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
            print("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
        # 输出USB接口的信息
        elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            print("\nu3v device: [%d]" % i)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                if per == 0:
                    break
                strModeName = strModeName + chr(per)
            print("device model name: %s" % strModeName)

            strSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                if per == 0:
                    break
                strSerialNumber = strSerialNumber + chr(per)
            print("user serial number: %s" % strSerialNumber)


def enable_device(nConnectionNum):
    """
    设备使能
    :param nConnectionNum: 设备编号
    :return: 相机, 图像缓存区, 图像数据大小
    """
    # ch:创建相机实例 | en:Creat Camera Object
    cam = MvCamera()

    # ch:选择设备并创建句柄 | en:Select device and create handle
    # cast(typ, val)，这个函数是为了检查val变量是typ类型的，但是这个cast函数不做检查，直接返回val
    stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

    ret = cam.MV_CC_CreateHandle(stDeviceList)
    if ret != 0:
        print("create handle fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:打开设备 | en:Open device
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print("open device fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
    if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
        nPacketSize = cam.MV_CC_GetOptimalPacketSize()
        if int(nPacketSize) > 0:
            ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
            if ret != 0:
                print("Warning: Set Packet Size fail! ret[0x%x]" % ret)
        else:
            print("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

    # ch:设置触发模式为off | en:Set trigger mode as off
    ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    if ret != 0:
        print("set trigger mode fail! ret[0x%x]" % ret)
        sys.exit()
    stFrameInfo = MV_FRAME_OUT_INFO_EX()
    # 从这开始，获取图片数据
    # ch:获取数据包大小 | en:Get payload size
    stParam = MVCC_INTVALUE()
    memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
    # MV_CC_GetIntValue，获取Integer属性值，handle [IN] 设备句柄
    # strKey [IN] 属性键值，如获取宽度信息则为"Width"
    # pIntValue [IN][OUT] 返回给调用者有关相机属性结构体指针
    # 得到图片尺寸，这一句很关键
    # payloadsize，为流通道上的每个图像传输的最大字节数，相机的PayloadSize的典型值是(宽x高x像素大小)，此时图像没有附加任何额外信息
    ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
    if ret != 0:
        print("get payload size fail! ret[0x%x]" % ret)
        sys.exit()

    nPayloadSize = stParam.nCurValue

    # ch:开始取流 | en:Start grab image
    ret = cam.MV_CC_StartGrabbing()


    if ret != 0:
        print("start grabbing fail! ret[0x%x]" % ret)
        sys.exit()
    #  返回获取图像缓存区。
    data_buf = (c_ubyte * nPayloadSize)()
    #  date_buf前面的转化不用，不然报错，因为转了是浮点型
    # rgb = cam.MV_CC_GetImageForRGB(stParam,nPayloadSize,byref(stFrameInfo),nMsec=1000)
    # cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    # cv2.imshow("image",rgb.pdata)
    return cam, data_buf, nPayloadSize,stFrameInfo


def Is_color_data(enGvspPixelType):
    if PixelType_Gvsp_BayerGR8 == enGvspPixelType or PixelType_Gvsp_BayerRG8 == enGvspPixelType \
        or PixelType_Gvsp_BayerGB8 == enGvspPixelType or PixelType_Gvsp_BayerBG8 == enGvspPixelType \
        or PixelType_Gvsp_BayerGR10 == enGvspPixelType or PixelType_Gvsp_BayerRG10 == enGvspPixelType \
        or PixelType_Gvsp_BayerGB10 == enGvspPixelType or PixelType_Gvsp_BayerBG10 == enGvspPixelType \
        or PixelType_Gvsp_BayerGR12 == enGvspPixelType or PixelType_Gvsp_BayerRG12 == enGvspPixelType \
        or PixelType_Gvsp_BayerGB12 == enGvspPixelType or PixelType_Gvsp_BayerBG12 == enGvspPixelType \
        or PixelType_Gvsp_BayerGR10_Packed == enGvspPixelType or PixelType_Gvsp_BayerRG10_Packed == enGvspPixelType \
        or PixelType_Gvsp_BayerGB10_Packed == enGvspPixelType or PixelType_Gvsp_BayerBG10_Packed == enGvspPixelType \
        or PixelType_Gvsp_BayerGR12_Packed == enGvspPixelType or PixelType_Gvsp_BayerRG12_Packed== enGvspPixelType \
        or PixelType_Gvsp_BayerGB12_Packed == enGvspPixelType or PixelType_Gvsp_BayerBG12_Packed == enGvspPixelType \
        or PixelType_Gvsp_YUV422_Packed == enGvspPixelType or PixelType_Gvsp_YUV422_YUYV_Packed == enGvspPixelType:
        return True
    else:
        return False

def Mono_numpy(data,nWidth,nHeight):
    data_ = np.frombuffer(data, count=int(nWidth * nHeight), dtype=np.uint8, offset=0)
    data_mono_arr = data_.reshape(nHeight, nWidth)
    numArray = np.zeros([nHeight, nWidth, 1],"uint8")
    numArray[:, :, 0] = data_mono_arr
    return numArray

def Color_numpy(data,nWidth,nHeight):
    data_ = np.frombuffer(data, count=int(nWidth*nHeight*3), dtype=np.uint8, offset=0)
    data_r = data_[0:nWidth*nHeight*3:3]
    data_g = data_[1:nWidth*nHeight*3:3]
    data_b = data_[2:nWidth*nHeight*3:3]

    data_r_arr = data_r.reshape(nHeight, nWidth)
    data_g_arr = data_g.reshape(nHeight, nWidth)
    data_b_arr = data_b.reshape(nHeight, nWidth)
    numArray = np.zeros([nHeight, nWidth, 3],"uint8")

    numArray[:, :, 2] = data_r_arr
    numArray[:, :, 1] = data_g_arr
    numArray[:, :, 0] = data_b_arr
    return numArray

def get_image(data_buf, nPayloadSize):
    """
    获取图像
    :param data_buf:
    :param nPayloadSize:
    :return: 图像
    """
    # 输出帧的信息
    stFrameInfo = MV_FRAME_OUT_INFO_EX()

    # void *memset(void *s, int ch, size_t n);
    # 函数解释:将s中当前位置后面的n个字节 (typedef unsigned int size_t )用 ch 替换并返回 s
    # memset:作用是在一段内存块中填充某个给定的值，它是对较大的结构体或数组进行清零操作的一种最快方法
    # byref(n)返回的相当于C的指针右值&n，本身没有被分配空间
    # 此处相当于将帧信息全部清空了
    #memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
    img_buff = None

    #ret = cam.MV_CC_GetOneFrameTimeout(byref(data_buf), nPayloadSize, stFrameInfo, 1000)

    ret = cam.MV_CC_GetOneFrameTimeout(byref(data_buf),nPayloadSize, stFrameInfo, 1000)
    if ret == 0:
        # 获取到图像的时间开始节点获取到图像的时间开始节点
        st_frame_info = stFrameInfo
        print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
        st_frame_info.nWidth, st_frame_info.nHeight, st_frame_info.nFrameNum))
        n_save_image_size = st_frame_info.nWidth * st_frame_info.nHeight * 3 + 2048
        if img_buff is None:
            img_buff = (c_ubyte * n_save_image_size)()

        # if True == b_save_jpg:
        #     Save_jpg()  # ch:保存Jpg图片 | en:Save Jpg
        # if buf_save_image is None:
        #     buf_save_image = (c_ubyte * n_save_image_size)()

        stParam = MV_SAVE_IMAGE_PARAM_EX()
        stParam.enImageType = MV_Image_Bmp;  # ch:需要保存的图像类型 | en:Image format to save
        stParam.enPixelType =st_frame_info.enPixelType  # ch:相机对应的像素格式 | en:Camera pixel type
        stParam.nWidth = st_frame_info.nWidth  # ch:相机对应的宽 | en:Width
        stParam.nHeight = st_frame_info.nHeight  # ch:相机对应的高 | en:Height
        stParam.nDataLen = st_frame_info.nFrameLen
        stParam.pData = cast(data_buf, POINTER(c_ubyte))
        #stParam.pImageBuffer = cast(byref(buf_save_image), POINTER(c_ubyte))
        stParam.nBufferSize = n_save_image_size  # ch:存储节点的大小 | en:Buffer node size
        stParam.nJpgQuality = 80;  # ch:jpg编码，仅在保存Jpg图像时有效。保存BMP时SDK内忽略该参数
        # if True == self.b_save_bmp:
        #     self.Save_Bmp()  # ch:保存Bmp图片 | en:Save Bmp
    # else:
    #     continue

    # 转换像素结构体赋值
    stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()
    memset(byref(stConvertParam), 0, sizeof(stConvertParam))
    stConvertParam.nWidth = st_frame_info.nWidth
    stConvertParam.nHeight = st_frame_info.nHeight
    stConvertParam.pSrcData = data_buf
    stConvertParam.nSrcDataLen = st_frame_info.nFrameLen
    stConvertParam.enSrcPixelType = st_frame_info.enPixelType

    # Mono8直接显示
    if PixelType_Gvsp_Mono8 ==st_frame_info.enPixelType:
        numArray = CameraOperation.Mono_numpy(data_buf, st_frame_info.nWidth,
                                              st_frame_info.nHeight)

    # RGB直接显示
    elif PixelType_Gvsp_RGB8_Packed == st_frame_info.enPixelType:
        numArray = cam.CameraOperation.Color_numpy(data_buf, st_frame_info.nWidth,
                                              st_frame_info.nHeight)

    # 如果是黑白且非Mono8则转为Mono8
    # elif True == self.Is_mono_data(self.st_frame_info.enPixelType):
    #     nConvertSize = self.st_frame_info.nWidth * self.st_frame_info.nHeight
    #     stConvertParam.enDstPixelType = PixelType_Gvsp_Mono8
    #     stConvertParam.pDstBuffer = (c_ubyte * nConvertSize)()
    #     stConvertParam.nDstBufferSize = nConvertSize
    #     ret = self.obj_cam.MV_CC_ConvertPixelType(stConvertParam)
    #     if ret != 0:
    #         tkinter.messagebox.showerror('show error', 'convert pixel fail! ret = ' + self.To_hex_str(ret))
    #         continue
    #     cdll.msvcrt.memcpy(byref(img_buff), stConvertParam.pDstBuffer, nConvertSize)
    #     numArray = CameraOperation.Mono_numpy(self, img_buff, self.st_frame_info.nWidth, self.st_frame_info.nHeight)

    # 如果是彩色且非RGB则转为RGB后显示
    elif True == Is_color_data(st_frame_info.enPixelType):
        nConvertSize =st_frame_info.nWidth * st_frame_info.nHeight * 3
        stConvertParam.enDstPixelType = PixelType_Gvsp_RGB8_Packed
        stConvertParam.pDstBuffer = (c_ubyte * nConvertSize)()
        stConvertParam.nDstBufferSize = nConvertSize
        ret = cam.MV_CC_ConvertPixelType(stConvertParam)
        # if ret != 0:
        #     tkinter.messagebox.showerror('show error', 'convert pixel fail! ret = ' + self.To_hex_str(ret))
        #     continue
        cdll.msvcrt.memcpy(byref(img_buff), stConvertParam.pDstBuffer, nConvertSize)
        numArray =Color_numpy( img_buff, st_frame_info.nWidth,
                                               st_frame_info.nHeight)

    image = np.array(numArray)  # 将c_ubyte_Array转化成ndarray得到（3686400，）
    print("222:",image.shape)

    image = image.reshape((2048, 3072,3))  # 根据自己分辨率进行转化



    import random
    #data = [random.randint(0, 1) for i in range(2048 * 2048)]

    # arr=image
    #
    # def rescale(arr):
    #     arr_min = arr.min()
    #     arr_max = arr.max()
    #     return (arr - arr_min) / (arr_max - arr_min)

    # arr[:, :, 0] = red_arr
    # arr[:, :, 1] = green_arr
    # arr[:, :, 2] = blue_arr
    # red_arr =arr[:, :, 0]
    # green_arr=arr[:, :, 1]
    # blue_arr=arr[:, :, 2]

    # arr = 255.0 * rescale(arr)
    #
    # arr = Image.fromarray(arr.astype(int), 'RGB')


    # img = Image.fromarray(arr, 'RGB')
    # img.show()
    return image


def close_device(cam, data_buf):
    """
    关闭设备
    :param cam:
    :param data_buf:
    """
    # ch:停止取流 | en:Stop grab image
    ret = cam.MV_CC_StopGrabbing()
    if ret != 0:
        print("stop grabbing fail! ret[0x%x]" % ret)
        del data_buf
        sys.exit()

    # ch:关闭设备 | Close device
    ret = cam.MV_CC_CloseDevice()
    if ret != 0:
        print("close deivce fail! ret[0x%x]" % ret)
        del data_buf
        sys.exit()

    # ch:销毁句柄 | Destroy handle
    ret = cam.MV_CC_DestroyHandle()
    if ret != 0:
        print("destroy handle fail! ret[0x%x]" % ret)
        del data_buf
        sys.exit()

    del data_buf


if __name__ == "__main__":
    # 获得设备信息
    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

    # ch: 枚举设备 | en:Enum device
    # nTLayerType[IN] 枚举传输层 ，pstDevList[OUT] 设备列表
    Enum_device(tlayerType, deviceList)

    # 获取相机和图像数据缓存区
    cam, data_buf, nPayloadSize,stFrameInfo = enable_device(0)  # 选择第一个设备
    print("111:",type(data_buf))
    #ret = cam.MV_CC_GetImageForRGB(data_buf, nPayloadSize, stFrameInfo, 1000)
    # ret=cam.MV_CC_GetImageBuffer(data_buf,nMsec=1000)

    while True:

        image = get_image(data_buf, nPayloadSize)

        #image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB) # 默认是BRG，要转化成RGB，颜色才正常
        #image3 = cv2.cvtColor(image2, cv2.COLOR_GRAY2RGBA)

        #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.imshow("image", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    # 关闭设备
    close_device(cam, data_buf)



