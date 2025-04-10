#dlib模型和diffusers批量生成
import torch
import cv2
import dlib
import numpy as np
from PIL import Image
from pathlib import Path
from diffusers import StableDiffusionDepth2ImgPipeline
import os


# 人脸检测异常类
class FaceDetectionError(Exception):
    pass


# 初始化人脸检测器
def init_face_detector(landmark_model_path="shape_predictor_68_face_landmarks.dat"):
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(str(landmark_model_path))
        return detector, predictor
    except Exception as e:
        raise RuntimeError(f"初始化人脸检测器失败: {str(e)}")


# 生成增强提示词
def generate_enhanced_prompt(original_prompt, landmarks, img_size):
    # 标准化坐标到0-1范围
    normalized_landmarks = [
        (p.x / img_size[0], p.y / img_size[1]) for p in landmarks.parts()
    ]

    # 构建结构化描述
    landmark_description = (
            "Facial landmarks coordinates (normalized): " +
            " ".join([f"({x:.3f},{y:.3f})" for x, y in normalized_landmarks])
    )

    # 添加解剖学描述
    anatomy_prompt = (
        "anatomically accurate facial features, "
        "professional portrait lighting, "
        "detailed skin texture with pores"
    )

    return f"{original_prompt} | {anatomy_prompt} | {landmark_description}"


# 主处理流程
def process_image(
        input_path,
        output_path,
        prompt="A picture of a men,natural facial features,no beard",
        negative_prompt="cartoon, low-quality, distorted",
        strength=0.7
):


    try:
        # 初始化模型
        detector, predictor = init_face_detector()
        pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-depth",
            torch_dtype=torch.float16
        ).to("cuda")

        # 加载并预处理图像
        init_image = Image.open(input_path).convert("RGB")
        img_size = init_image.size
        cv_image = cv2.cvtColor(np.array(init_image), cv2.COLOR_RGB2BGR)


        # 人脸检测
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if not faces:
            raise FaceDetectionError("未检测到人脸")

        # 关键点检测
        landmarks = predictor(gray, faces[0])

        # 生成增强提示词
        enhanced_prompt = generate_enhanced_prompt(prompt, landmarks, img_size)

        # 图像生成
        result = pipe(
            prompt=enhanced_prompt,
            image=init_image,
            negative_prompt=negative_prompt,
            strength=strength,
            guidance_scale=12,
            num_inference_steps=45,
            controlnet_conditioning_scale=0.8,
            generator=torch.Generator(device="cuda").manual_seed(42)
        )

        # 保存结果
        result.images[0].save(output_path)
        print(f"图像已保存至 {output_path}")

    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        raise


if __name__ == "__main__":
    # 配置参数

    input_folder = "D:\\pythonproject\\PythonProject1\\data\\input"  # 输入文件夹路径
    output_folder = "D:\\pythonproject\\PythonProject1\\data\\output"  # 输出文件夹路径


    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有图片
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)  # 保持文件名一致

            # 执行处理
            process_image(
                input_path=input_path,
                output_path=output_path,
                # prompt="A man with a pained facial expression, showing distress and discomfort,"
                #        "slightly squinted eyes with visible crow's feet,furrowed eyebrows creating deep forehead wrinkles,"
                #        "partially open mouth with tense lip muscles,"
                #        "porcelain skin with subtle pores and natural oil highlights",
                prompt=
                       "A picture of a men,natural facial features",
                       # "A picture of an old women,",
                       # "A close-up portrait of a men with pain intensity description",
                       #  "A man with a pained facial expression, showing distress and discomfort,"
                       # "A close-up picture of a person with a highly detailed facial expression showing a specific level of pain,"
                       # "emphasizing the eyes, eyebrows, and mouth to convey the exact intensity of pain,"
                       # "realistic skin texture with pores and subtle wrinkle,",
                       # "natural skin texture with pores",



                negative_prompt=(
                    "cartoon, low-quality, distorted, "
                    "unnatural facial features, "
                    "asymmetrical eyes, "
                    "unrealistic skin texture"
                ),
                strength=0.7,
            )