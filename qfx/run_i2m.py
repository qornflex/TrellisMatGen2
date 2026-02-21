import sys
import os
import cv2
import json
import math
import pathlib
import warnings
import bpy
import shutil
import pygltflib
from pygltflib.utils import ImageFormat
import numpy as np

import imageio
from PIL import Image, ImageFilter

import utils.imgops as ops
import utils.architecture.architecture as arch

import torch
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel

# ======================================================================================================================

warnings.filterwarnings('ignore')

NORMAL_MAP_MODEL = 'matgen/utils/models/1x_NormalMapGenerator-CX-Lite_200000_G.pth'
OTHER_MAP_MODEL = 'matgen/utils/models/1x_FrankenMapGenerator-CX-Lite_215000_G.pth'

MATGEN_MODELS = []

DEVICE = torch.device('cuda')

# ======================================================================================================================

def process(img, model):
    global DEVICE
    img = img * 1. / np.iinfo(img.dtype).max
    img = img[:, :, [2, 1, 0]]
    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(DEVICE)

    output = model(img_LR).data.squeeze(
        0).float().cpu().clamp_(0, 1).numpy()
    output = output[[2, 1, 0], :, :]
    output = np.transpose(output, (1, 2, 0))
    output = (output * 255.).round()
    return output


def load_matgen_model(model_path):
    global DEVICE
    state_dict = torch.load(model_path)
    model = arch.RRDB_Net(3, 3, 32, 12, gc=32, upscale=1, norm_type=None, act_type='leakyrelu',
                            mode='CNA', res_scale=1, upsample_mode='upconv')
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    return model.to(DEVICE)


def extract_textures(input_filepath):

    texture_filename = pathlib.Path(input_filepath).stem
    input_folder = os.path.dirname(input_filepath)

    texture_folder = f"{input_folder}/textures"

    if not os.path.exists(texture_folder):
        os.mkdir(texture_folder)

    # export textures

    gltf = pygltflib.GLTF2()
    gltf.draco_compression = False  # skip Draco

    glb = gltf.load(input_filepath)
    if len(glb.images) > 0:

        glb.convert_images(ImageFormat.FILE, texture_folder, override=True)

        base_color_filepath = os.path.join(texture_folder, f"{texture_filename}_BaseColor.png")
        init_orm_filepath = os.path.join(texture_folder, f"{texture_filename}_InitORM.png")

        for i in range(0, len(glb.images)):

            img_filepath = f'{texture_folder}/{glb.images[i].uri}'

            if i == 0:

                if os.path.exists(base_color_filepath):
                    os.remove(base_color_filepath)
                os.rename(img_filepath, base_color_filepath)

            if i == 1:

                if os.path.exists(init_orm_filepath):
                    os.remove(init_orm_filepath)
                os.rename(img_filepath, init_orm_filepath)

        # Load InitORM image
        img = Image.open(init_orm_filepath).convert("RGB")

        # Split channels
        r, g, b = img.split()

        # Save channels
        r.save(os.path.join(texture_folder, f"{texture_filename}_AO.png"))
        g.save(os.path.join(texture_folder, f"{texture_filename}_Roughness.png"))
        b.save(os.path.join(texture_folder, f"{texture_filename}_Metallic.png"))

        os.remove(init_orm_filepath)


def generate_ao_map(displacement_path: str, output_path: str = None, blur_radius: int = 2):
    """
    Generates a rough Ambient Occlusion (AO) map from a displacement (height) map.

    Args:
        displacement_path (str): Path to the displacement (height) map PNG.
        output_path (str, optional): Path to save the AO map.
                                     If None, saves alongside input with '_AO' suffix.
        blur_radius (int): Radius for Gaussian blur to simulate soft occlusion.

    Returns:
        str: Path of the saved AO map.
    """
    # Load displacement map as grayscale
    disp_img = Image.open(displacement_path).convert("L")
    disp_arr = np.array(disp_img, dtype=np.float32) / 255.0  # normalize 0-1

    # Invert height: high areas = less occluded
    inv_height = 1.0 - disp_arr

    # Convert back to image
    inv_height_img = Image.fromarray((inv_height * 255).astype(np.uint8))

    # Apply Gaussian blur to simulate ambient occlusion
    ao_img = inv_height_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Determine output path
    if output_path is None:
        base, ext = os.path.splitext(displacement_path)
        output_path = f"{base}_AO.png"

    # Save AO map
    ao_img.save(output_path)
    print(f"AO map saved to: {output_path}")
    return output_path


def generate_orm_map(ao_path=None, roughness_path=None, metallic_path=None, output_path=None):
    """
    Combines AO, Roughness, and Metallic textures into a single ORM map (R=AO, G=Roughness, B=Metallic).

    Args:
        ao_path (str): Path to AO texture (grayscale)
        roughness_path (str): Path to Roughness texture (grayscale)
        metallic_path (str): Path to Metallic texture (grayscale)
        output_path (str): Path to save the ORM texture. If None, saves as 'ORM.png' in the first input folder.

    Returns:
        str: Path of the saved ORM map.
    """

    # Load textures or create default gray if missing
    def load_gray(path, size=None):
        if path and os.path.exists(path):
            img = Image.open(path).convert("L")
        else:
            img = Image.new("L", size if size else (1024, 1024), 255)
        return img

    # Determine base size from first available texture
    base_size = None
    for path in [ao_path, roughness_path, metallic_path]:
        if path and os.path.exists(path):
            base_size = Image.open(path).size
            break
    if base_size is None:
        base_size = (1024, 1024)  # default

    ao_img = load_gray(ao_path, base_size)
    rough_img = load_gray(roughness_path, base_size)
    metal_img = load_gray(metallic_path, base_size)

    # Merge into RGB
    orm_img = Image.merge("RGB", (ao_img, rough_img, metal_img))

    # Determine output path
    if not output_path:
        first_path = next((p for p in [ao_path, roughness_path, metallic_path] if p), None)
        folder = os.path.dirname(first_path) if first_path else os.getcwd()
        output_path = os.path.join(folder, "ORM.png")

    orm_img.save(output_path)
    print(f"ORM map saved to: {output_path}")
    return output_path


def generate_pbr(albedo_filepath,
                 override=False,
                 tile_size=512,
                 seamless=False,
                 mirror=False,
                 replicate=False):

    global MATGEN_MODELS

    basename = os.path.splitext(os.path.basename(albedo_filepath))[0]
    basename = basename.replace("_BaseColor", "")

    output_folder = os.path.dirname(albedo_filepath)

    # read image
    try:
        img = cv2.imread(albedo_filepath, cv2.cv2.IMREAD_COLOR)
    except:
        img = cv2.imread(albedo_filepath, cv2.IMREAD_COLOR)

    # Seamless modes
    if seamless:
        img = cv2.copyMakeBorder(img, 16, 16, 16, 16, cv2.BORDER_WRAP)
    elif mirror:
        img = cv2.copyMakeBorder(img, 16, 16, 16, 16, cv2.BORDER_REFLECT_101)
    elif replicate:
        img = cv2.copyMakeBorder(img, 16, 16, 16, 16, cv2.BORDER_REPLICATE)

    img_height, img_width = img.shape[:2]

    # Whether to perform the split/merge action
    do_split = img_height > tile_size or img_width > tile_size

    if do_split:
        rlts = ops.esrgan_launcher_split_merge(img, process, MATGEN_MODELS, scale_factor=1, tile_size=tile_size)
    else:
        rlts = [process(img, model) for model in MATGEN_MODELS]

    if seamless or mirror or replicate:
        rlts = [ops.crop_seamless(rlt) for rlt in rlts]

    normal_map = rlts[0]
    roughness = rlts[1][:, :, 1]
    displacement = rlts[1][:, :, 0]

    normal_name = '{:s}_Normal.png'.format(basename)
    cv2.imwrite(os.path.join(output_folder, normal_name), normal_map)

    # rough_name = '{:s}_Roughness.png'.format(basename)
    # rough_img = roughness
    # cv2.imwrite(os.path.join(output_folder, rough_name), rough_img)

    displace_name = '{:s}_Displacement.png'.format(basename)
    cv2.imwrite(os.path.join(output_folder, displace_name), displacement)

    metallic_file = albedo_filepath.replace("_BaseColor", "_Metallic")

    # generate_metallic_map(albedo_filepath, metallic_file)

    displace_file = albedo_filepath.replace("_BaseColor", "_Displacement")
    ao_file = albedo_filepath.replace("_BaseColor", "_AO")
    generate_ao_map(displace_file, ao_file)

    roughness_file = albedo_filepath.replace("_BaseColor", "_Roughness")
    orm_file = albedo_filepath.replace("_BaseColor", "_ORM")

    generate_orm_map(ao_file, roughness_file, metallic_file, orm_file)


def fix_mesh(input_filepath, mesh_scale=1.0, smooth_normals=True, smooth_normals_angle=30.0):

    input_dir = os.path.dirname(input_filepath)

    glb_file_name = pathlib.Path(input_filepath).stem

    # Texture paths
    albedo_texture_filepath = os.path.join(input_dir, "textures", f"{glb_file_name}_basecolor.png")
    normal_texture_filepath = os.path.join(input_dir, "textures", f"{glb_file_name}_normal.png")
    roughness_texture_filepath = os.path.join(input_dir, "textures", f"{glb_file_name}_roughness.png")
    metallic_texture_filepath = os.path.join(input_dir, "textures", f"{glb_file_name}_metallic.png")

    # Clear existing scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import GLB
    input_filepath = os.path.abspath(input_filepath)
    if not os.path.exists(input_filepath):
        raise FileNotFoundError(f"GLB file not found: {input_filepath}")

    bpy.ops.import_scene.gltf(filepath=input_filepath)

    # Remove 'world' dummy parent if it exists
    for obj in bpy.data.objects:
        if obj.type == 'EMPTY' and obj.name.lower() == 'world':
            bpy.data.objects.remove(obj, do_unlink=True)

    # Process all mesh objects
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.parent = None
            obj.name = glb_file_name
            obj.data.name = glb_file_name
            bpy.context.view_layer.objects.active = obj

            # Apply smoothing
            if smooth_normals:
                mesh = obj.data
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.shade_smooth()  # ← This actually smooths the faces visually
                mesh.use_auto_smooth = True
                mesh.auto_smooth_angle = smooth_normals_angle * (3.14159 / 180)
                mesh.update()

            # Apply scale
            obj.scale = (mesh_scale, mesh_scale, mesh_scale)
            bpy.context.view_layer.update()

            # Apply transformations
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

            # Assign textures
            assign_textures(
                obj,
                albedo_path=albedo_texture_filepath,
                normal_path=normal_texture_filepath,
                roughness_path=roughness_texture_filepath,
                metallic_path=metallic_texture_filepath,
            )

    textures_folder = os.path.join(input_dir, "textures")
    redirect_textures_to_folder(textures_folder)

    # Resave GLB

    bpy.ops.export_scene.gltf(
        filepath=os.path.splitext(input_filepath)[0] + ".glb",
        export_format='GLB',
        export_texture_dir="textures"
    )

    # Export to FBX

    bpy.ops.export_scene.fbx(
        filepath=os.path.splitext(input_filepath)[0] + ".fbx",
        use_selection=False,
        apply_unit_scale=True,
        apply_scale_options='FBX_SCALE_ALL',
        axis_forward='-Z',
        axis_up='Y',
        object_types={'MESH'},
        # mesh_smooth_type='FACE',
        mesh_smooth_type='EDGE',
        path_mode='COPY',
        embed_textures=False
    )

    # Remove unwanted .fbm folder if Blender created it
    fbm_folder = os.path.splitext(input_filepath)[0] + ".fbm"
    if os.path.exists(fbm_folder):
        shutil.rmtree(fbm_folder, ignore_errors=True)


def redirect_textures_to_folder(textures_folder):
    for image in bpy.data.images:
        if image and image.filepath:
            tex_name = os.path.basename(image.filepath)
            new_path = os.path.join(textures_folder, tex_name)
            image.filepath = bpy.path.abspath(new_path)
            image.reload()


def assign_textures(obj, albedo_path=None, normal_path=None, roughness_path=None, metallic_path=None):

    """Assign PBR textures to the object's material(s), supporting combined or separate metallic/roughness."""

    for mat_slot in obj.material_slots:
        mat = mat_slot.material
        if not mat or not mat.node_tree:
            continue

        mat.name = obj.name

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        for node in nodes:
            if node.type in {'TEX_IMAGE', 'NORMAL_MAP', 'SEPARATE_RGB'}:
                nodes.remove(node)

        # Find Principled BSDF
        bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
        if not bsdf:
            continue

        # --- Albedo ---
        if albedo_path and os.path.exists(albedo_path):
            img_node = nodes.new('ShaderNodeTexImage')
            img_node.image = bpy.data.images.load(albedo_path)
            img_node.image.colorspace_settings.is_data = False  # sRGB
            img_node.location = (-600, 300)
            links.new(img_node.outputs['Color'], bsdf.inputs['Base Color'])

        # --- Normal Map ---
        if normal_path and os.path.exists(normal_path):
            normal_img_node = nodes.new('ShaderNodeTexImage')
            normal_img_node.image = bpy.data.images.load(normal_path)
            normal_img_node.image.colorspace_settings.is_data = True  # Non-color
            normal_img_node.location = (-600, 100)

            normal_map_node = nodes.new('ShaderNodeNormalMap')
            normal_map_node.location = (-400, 100)

            links.new(normal_img_node.outputs['Color'], normal_map_node.inputs['Color'])
            links.new(normal_map_node.outputs['Normal'], bsdf.inputs['Normal'])

        # --- Roughness ---
        if roughness_path and os.path.exists(roughness_path):
            rough_node = nodes.new('ShaderNodeTexImage')
            rough_node.image = bpy.data.images.load(roughness_path)
            rough_node.image.colorspace_settings.is_data = True  # Non-color
            rough_node.location = (-600, -100)
            links.new(rough_node.outputs['Color'], bsdf.inputs['Roughness'])

        # --- Metallic ---
        if metallic_path and os.path.exists(metallic_path):
            metal_node = nodes.new('ShaderNodeTexImage')
            metal_node.image = bpy.data.images.load(metallic_path)
            metal_node.image.colorspace_settings.is_data = True  # Non-color
            metal_node.location = (-600, -300)
            links.new(metal_node.outputs['Color'], bsdf.inputs['Metallic'])


def generate(input_filelist):

    # Load MatGen Models

    global MATGEN_MODELS

    print("[MATGEN] Loading models...")

    MATGEN_MODELS = [
        load_matgen_model(NORMAL_MAP_MODEL),  # NORMAL MAP
        load_matgen_model(OTHER_MAP_MODEL)  # ROUGHNESS/DISPLACEMENT MAPS
    ]

    # Load JSON
    with open(input_filelist, "r", encoding="utf-8") as f:
        data = json.load(f)

    params = data["parameters"]

    seed = params["seed"]
    texture_size = params["texture_size"]
    decimation_target = params["decimation_target"]

    output_folder = params["output_folder"]

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # 1. Setup Environment Map
    # envmap = EnvMap(torch.tensor(
    #     cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
    #     dtype=torch.float32, device='cuda'
    # ))

    # 2. Load Pipeline
    model_path = "./models/microsoft/TRELLIS.2-4B"

    if not os.path.exists(model_path):
        print(f"Can't find TRELLIS model ({model_path})")
        return

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_path)
    pipeline.cuda()

    inputs = data["inputs"]

    count = 0
    total = len(inputs)

    for input in inputs:

        count += 1

        progress = math.floor((float(count) / float(total)) * 100.0)

        print(f"TrellisProgress: {progress}%")

        # Load Image & Run

        if os.path.exists(input["filepath"]):

            mesh_name = input["name"]

            output_mesh_folder = f"{output_folder}/{mesh_name}"
            if not os.path.exists(output_mesh_folder):
                os.makedirs(output_mesh_folder)

            image = Image.open(input["filepath"])
            mesh = pipeline.run(image=image,
                                num_samples=1,
                                seed=seed)[0]

            mesh.simplify(16777216) # nvdiffrast limit

            # 4. Render Video
            # video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
            # imageio.mimsave(f"{output_folder}/{mesh_name}.mp4", video, fps=15)

            # 5. Export to GLB
            glb = o_voxel.postprocess.to_glb(
                vertices            =   mesh.vertices,
                faces               =   mesh.faces,
                attr_volume         =   mesh.attrs,
                coords              =   mesh.coords,
                attr_layout         =   mesh.layout,
                voxel_size          =   mesh.voxel_size,
                aabb                =   [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target   =   decimation_target,
                texture_size        =   texture_size,
                remesh              =   True,
                remesh_band         =   1,
                remesh_project      =   0,
                verbose             =   True
            )

            glb_filepath = f"{output_mesh_folder}/{mesh_name}.glb"

            glb.export(glb_filepath, extension_webp=False)

            extract_textures(glb_filepath)

            print("Generating PBR Textures: 100%")

            albedo_filepath = f"{output_mesh_folder}/textures/{mesh_name}_BaseColor.png"
            generate_pbr(albedo_filepath, override=True)

            print("Finalizing Mesh: 100%")

            mesh_scale = params["mesh_scale"]
            smooth_normals = params["smooth_normals"]
            smooth_normals_angle = params["smooth_normals_angle"]

            fix_mesh(glb_filepath, mesh_scale, smooth_normals, smooth_normals_angle)



if __name__ == "__main__":

    in_input_filelist = sys.argv[-1]

    generate(input_filelist=in_input_filelist)