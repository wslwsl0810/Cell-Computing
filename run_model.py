#输出数量
# import numpy as np
# import os, glob, cv2, torch
# import pytorch_lightning as pl
# import segmentation_models_pytorch as smp
# import torchvision.transforms as transforms
# import skimage.measure as skm
# from tqdm import tqdm

# class SegModel(pl.LightningModule):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.net = smp.Unet(encoder_weights=None, classes=2, encoder_name='resnet50')

#     def forward(self, x):
#         out = self.net(x)
#         out = torch.argmax(out, dim=1, keepdim=True).long()
#         return out

#     def predict(self, path):
#         img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)  # todo move to function needed more often best would be to have it in the model as predict with the path
#         preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet50')
#         img = transforms.ToTensor()(preprocessing_fn(img)).float()
#         img = torch.unsqueeze(img, 0)
#         r = self(img)
#         processed = torch.argmax(torch.squeeze(r.detach()), dim=0).long().numpy()  # process to masks
#         labels = skm.label(processed).transpose((1, 0)) # todo why strange transpose
#         return labels


# def run_model(img_path, model, min_area=50):
#     """
#     Runs the model on patches of the image, concats the predictions together,
#     and counts the number of white cells.
#     :param img_path: path to the image
#     :param model: segmentation model with the forward path specified
#     :param min_area: minimum area threshold to filter small noise regions
#     :return: tuple (numpy array of predictions with values 0 and 255, number of white cells)
#     """
#     img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
#     img_pad = np.zeros((img.shape[0] + 224, img.shape[1] + 224, img.shape[2]))
#     label_pad = np.zeros((img.shape[0] + 224, img.shape[1] + 224))
#     img_pad[112:-112, 112:-112] = img
#     imgs = []
#     preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet50')

#     for i in range(img_pad.shape[0] // 800 + 1):  # split image into 1024*1024 chunks
#         for j in range(img_pad.shape[1] // 800 + 1):
#             x, x1, y, y1 = 800 * i, 800 * i + 1024, 800 * j, 800 * j + 1024
#             to_much_x = x1 - img_pad.shape[0]
#             to_much_y = y1 - img_pad.shape[1]
#             if to_much_x > 0:
#                 x = img_pad.shape[0] - 1024
#                 x1 = img_pad.shape[0]
#             if to_much_y > 0:
#                 y = img_pad.shape[1] - 1024
#                 y1 = img_pad.shape[1]
#             input_img = img_pad[x:x1, y:y1].copy()
#             img1 = preprocessing_fn(input_img)
#             img1 = transforms.ToTensor()(img1).float()
#             img1 = torch.unsqueeze(img1, 0)
#             r = model.forward(img1)
#             processed = torch.squeeze(r.detach()).long().numpy()  # process to masks
#             labels = skm.label(processed).transpose((1, 0))  # label connected components
#             result_semseg = (labels.T > 0).astype(int) * 255
#             label_pad[x + 112:x1 - 112, y + 112:y1 - 112] = result_semseg[112:-112, 112:-112]

#     # Crop back to original size
#     final_mask = label_pad[112:-112, 112:-112]

#     # Count white cells (connected components)
#     binary_mask = (final_mask > 0).astype(np.uint8)  # Convert to binary (0 and 1)
#     labeled_mask, num_cells = skm.label(binary_mask, return_num=True)  # Count distinct regions

#     # Optional: Filter small regions (noise)
#     if min_area > 0:
#         props = skm.regionprops(labeled_mask)
#         filtered_mask = np.zeros_like(labeled_mask)
#         valid_count = 0
#         for prop in props:
#             if prop.area >= min_area:  # Keep only regions larger than min_area
#                 filtered_mask[labeled_mask == prop.label] = 1
#                 valid_count += 1
#         num_cells = valid_count
#         final_mask = filtered_mask * 255  # Update mask if filtering applied

#     return final_mask, num_cells


# if __name__ == '__main__':
#     import argparse, os

#     parser = argparse.ArgumentParser(description='Segment image and count white cells')
#     parser.add_argument('-m', '--model', default='./final_semseg_coniferen_model.pth',
#                         help='model with pretrained weights')
#     parser.add_argument('-i', '--input', default='./input', help='path to the input file')
#     parser.add_argument('-o', '--output', default='./output', help='path where the output file should be stored')
#     args = parser.parse_args()
#     assert os.path.exists(args.model), 'model path does not exist'
#     assert os.path.exists(args.input), 'input path does not exist'

#     ### Prepare Model ###
#     print('load model')
#     model = SegModel()
#     model.load_state_dict(torch.load(args.model)['state_dict'])
#     model.eval()

#     ### Process file ###
#     print('run model')
#     prediction, cell_count = run_model(args.input, model, min_area=50)  # min_area 可调整
#     print(f'Number of white cells detected: {cell_count}')

#     ### Save output ###
#     print('save output')
#     os.makedirs(os.path.dirname(args.output), exist_ok=True)
#     cv2.imwrite(f"{args.output}", prediction)
#     print('finished successfully')



#计数并标记数字
# import numpy as np
# import os, glob, cv2, torch
# import pytorch_lightning as pl
# import segmentation_models_pytorch as smp
# import torchvision.transforms as transforms
# import skimage.measure as skm
# from tqdm import tqdm

# class SegModel(pl.LightningModule):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.net = smp.Unet(encoder_weights=None, classes=2, encoder_name='resnet50')

#     def forward(self, x):
#         out = self.net(x)
#         out = torch.argmax(out, dim=1, keepdim=True).long()
#         return out

#     def predict(self, path):
#         img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
#         preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet50')
#         img = transforms.ToTensor()(preprocessing_fn(img)).float()
#         img = torch.unsqueeze(img, 0)
#         r = self(img)
#         processed = torch.argmax(torch.squeeze(r.detach()), dim=0).long().numpy()
#         labels = skm.label(processed).transpose((1, 0))
#         return labels

# def run_model(img_path, model, min_area=50):
#     """
#     Runs the model on patches of the image, concats the predictions together,
#     counts the number of white cells, and labels each cell with a number on both
#     the binary mask and the original colored image. Adjusts label positions and sizes to avoid edge clipping.
#     :param img_path: path to the image
#     :param model: segmentation model with the forward path specified
#     :param min_area: minimum area threshold to filter small noise regions
#     :return: tuple (binary mask, number of white cells, labeled binary image, labeled colored image)
#     """
#     img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
#     img_height, img_width = img.shape[:2]  # Get image dimensions
#     img_pad = np.zeros((img.shape[0] + 224, img.shape[1] + 224, img.shape[2]))
#     label_pad = np.zeros((img.shape[0] + 224, img.shape[1] + 224))
#     img_pad[112:-112, 112:-112] = img
#     preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet50')

#     for i in range(img_pad.shape[0] // 800 + 1):
#         for j in range(img_pad.shape[1] // 800 + 1):
#             x, x1, y, y1 = 800 * i, 800 * i + 1024, 800 * j, 800 * j + 1024
#             to_much_x = x1 - img_pad.shape[0]
#             to_much_y = y1 - img_pad.shape[1]
#             if to_much_x > 0:
#                 x = img_pad.shape[0] - 1024
#                 x1 = img_pad.shape[0]
#             if to_much_y > 0:
#                 y = img_pad.shape[1] - 1024
#                 y1 = img_pad.shape[1]
#             input_img = img_pad[x:x1, y:y1].copy()
#             img1 = preprocessing_fn(input_img)
#             img1 = transforms.ToTensor()(img1).float()
#             img1 = torch.unsqueeze(img1, 0)
#             r = model.forward(img1)
#             processed = torch.squeeze(r.detach()).long().numpy()
#             labels = skm.label(processed).transpose((1, 0))
#             result_semseg = (labels.T > 0).astype(int) * 255
#             label_pad[x + 112:x1 - 112, y + 112:y1 - 112] = result_semseg[112:-112, 112:-112]

#     # Crop back to original size
#     final_mask = label_pad[112:-112, 112:-112]

#     # Count white cells (connected components)
#     binary_mask = (final_mask > 0).astype(np.uint8)
#     labeled_mask, num_cells = skm.label(binary_mask, return_num=True)

#     # Filter small regions and create labeled images
#     if min_area > 0:
#         props = skm.regionprops(labeled_mask)
#         filtered_mask = np.zeros_like(labeled_mask)
#         valid_count = 0
#         labeled_image = final_mask.copy()  # Labeled binary image
#         colored_labeled_image = img.copy()  # Labeled colored image
#         for prop in props:
#             if prop.area >= min_area:
#                 filtered_mask[labeled_mask == prop.label] = 1
#                 valid_count += 1
#                 centroid = prop.centroid  # (row, col)

#                 # Dynamic font scale based on cell area
#                 font_scale = max(0.1, min(0.5, np.sqrt(prop.area) / 50))
#                 font_thickness = 1 if prop.area > 200 else 0

#                 # Draw number on the binary mask
#                 cv2.putText(
#                     labeled_image,
#                     str(valid_count),
#                     (int(centroid[1]), int(centroid[0])),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     font_scale,
#                     0,
#                     font_thickness
#                 )

#                 # Draw number on the colored image with a background rectangle
#                 text = str(valid_count)
#                 x, y = int(centroid[1]), int(centroid[0])
#                 padding = 4

#                 # Iteratively adjust font size and position to fit within image bounds
#                 max_iterations = 10
#                 iteration = 0
#                 while iteration < max_iterations:
#                     text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
#                     text_w, text_h = text_size
#                     rect_top_left = (x - padding, y - text_h - padding)
#                     rect_bottom_right = (x + text_w + padding, y + padding)

#                     # Check if the text fits within image bounds
#                     fits = True
#                     if rect_top_left[0] < 0:  # Left edge
#                         x -= rect_top_left[0]  # Move right
#                         fits = False
#                     if rect_bottom_right[0] > img_width:  # Right edge
#                         x -= (rect_bottom_right[0] - img_width)  # Move left
#                         fits = False
#                     if rect_top_left[1] < 0:  # Top edge
#                         y -= rect_top_left[1]  # Move down
#                         fits = False
#                     if rect_bottom_right[1] > img_height:  # Bottom edge
#                         y -= (rect_bottom_right[1] - img_height)  # Move up
#                         fits = False

#                     if fits:
#                         break

#                     # If the text still doesn't fit, reduce font size
#                     font_scale *= 0.8
#                     if font_scale < 0.05:  # Minimum font size
#                         font_scale = 0.05
#                         break
#                     iteration += 1

#                 # Final position and size after adjustments
#                 text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
#                 text_w, text_h = text_size
#                 rect_top_left = (x - padding, y - text_h - padding)
#                 rect_bottom_right = (x + text_w + padding, y + padding)

#                 # Draw background rectangle
#                 cv2.rectangle(
#                     colored_labeled_image,
#                     rect_top_left,
#                     rect_bottom_right,
#                     (0, 0, 0),
#                     -1
#                 )

#                 # Draw text at adjusted position
#                 cv2.putText(
#                     colored_labeled_image,
#                     text,
#                     (x, y),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     font_scale,
#                     (255, 255, 255),
#                     font_thickness
#                 )
#         num_cells = valid_count
#         final_mask = filtered_mask * 255

#     return final_mask, num_cells, labeled_image, colored_labeled_image

# if __name__ == '__main__':
#     import argparse, os

#     parser = argparse.ArgumentParser(description='Segment image and count white cells')
#     parser.add_argument('-m', '--model', default='./final_semseg_coniferen_model.pth',
#                         help='model with pretrained weights')
#     parser.add_argument('-i', '--input', default='./input', help='path to the input file')
#     parser.add_argument('-o', '--output', default='./output/output.png', help='path where the output file should be stored')
#     args = parser.parse_args()
#     assert os.path.exists(args.model), 'model path does not exist'
#     assert os.path.exists(args.input), 'input path does not exist'

#     ### Normalize output path
#     if not args.output.endswith('.png'):
#         args.output = args.output + '.png'

#     ### Prepare Model ###
#     print('load model')
#     model = SegModel()
#     model.load_state_dict(torch.load(args.model)['state_dict'])
#     model.eval()

#     ### Process file ###
#     print('run model')
#     prediction, cell_count, labeled_image, colored_labeled_image = run_model(args.input, model, min_area=50)
#     print(f'Number of white cells detected: {cell_count}')

#     ### Save output ###
#     print('save output')
#     os.makedirs(os.path.dirname(args.output), exist_ok=True)
#     # Save the original binary mask
#     success = cv2.imwrite(args.output, prediction)
#     if not success:
#         print(f"Failed to save binary mask to {args.output}")
#     else:
#         print(f"Binary mask saved as: {args.output}")
#     # Save the labeled binary image with numbers
#     labeled_output = args.output.replace('.png', '_labeled.png')
#     success = cv2.imwrite(labeled_output, labeled_image)
#     if not success:
#         print(f"Failed to save labeled binary image to {labeled_output}")
#     else:
#         print(f"Labeled binary image saved as: {labeled_output}")
#     # Save the labeled colored image with numbers
#     print(f"colored_labeled_image shape: {colored_labeled_image.shape}, dtype: {colored_labeled_image.dtype}")
#     colored_labeled_image = colored_labeled_image.astype(np.uint8)
#     colored_labeled_output = args.output.replace('.png', '_colored_labeled.png')
#     success = cv2.imwrite(colored_labeled_output, colored_labeled_image)
#     if not success:
#         print(f"Failed to save colored labeled image to {colored_labeled_output}")
#     else:
#         print(f"Labeled colored image saved as: {colored_labeled_output}")
#     print('finished successfully')

#计算细胞面积
# import numpy as np
# import os, glob, cv2, torch
# import pytorch_lightning as pl
# import segmentation_models_pytorch as smp
# import torchvision.transforms as transforms
# import skimage.measure as skm
# from tqdm import tqdm

# class SegModel(pl.LightningModule):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.net = smp.Unet(encoder_weights=None, classes=2, encoder_name='resnet50')

#     def forward(self, x):
#         out = self.net(x)
#         out = torch.argmax(out, dim=1, keepdim=True).long()
#         return out

#     def predict(self, path):
#         img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
#         preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet50')
#         img = transforms.ToTensor()(preprocessing_fn(img)).float()
#         img = torch.unsqueeze(img, 0)
#         r = self(img)
#         processed = torch.argmax(torch.squeeze(r.detach()), dim=0).long().numpy()
#         labels = skm.label(processed).transpose((1, 0))
#         return labels

# def run_model(img_path, model, min_area=50):
#     """
#     Runs the model on patches of the image, concats the predictions together,
#     counts the number of white cells, labels each cell with a number, and returns cell areas.
#     :param img_path: path to the image
#     :param model: segmentation model with the forward path specified
#     :param min_area: minimum area threshold to filter small noise regions
#     :return: tuple (binary mask, number of white cells, labeled binary image, labeled colored image, cell areas)
#     """
#     img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
#     img_height, img_width = img.shape[:2]
#     img_pad = np.zeros((img.shape[0] + 224, img.shape[1] + 224, img.shape[2]))
#     label_pad = np.zeros((img.shape[0] + 224, img.shape[1] + 224))
#     img_pad[112:-112, 112:-112] = img
#     preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet50')

#     for i in range(img_pad.shape[0] // 800 + 1):
#         for j in range(img_pad.shape[1] // 800 + 1):
#             x, x1, y, y1 = 800 * i, 800 * i + 1024, 800 * j, 800 * j + 1024
#             to_much_x = x1 - img_pad.shape[0]
#             to_much_y = y1 - img_pad.shape[1]
#             if to_much_x > 0:
#                 x = img_pad.shape[0] - 1024
#                 x1 = img_pad.shape[0]
#             if to_much_y > 0:
#                 y = img_pad.shape[1] - 1024
#                 y1 = img_pad.shape[1]
#             input_img = img_pad[x:x1, y:y1].copy()
#             img1 = preprocessing_fn(input_img)
#             img1 = transforms.ToTensor()(img1).float()
#             img1 = torch.unsqueeze(img1, 0)
#             r = model.forward(img1)
#             processed = torch.squeeze(r.detach()).long().numpy()
#             labels = skm.label(processed).transpose((1, 0))
#             result_semseg = (labels.T > 0).astype(int) * 255
#             label_pad[x + 112:x1 - 112, y + 112:y1 - 112] = result_semseg[112:-112, 112:-112]

#     # Crop back to original size
#     final_mask = label_pad[112:-112, 112:-112]

#     # Count white cells (connected components)
#     binary_mask = (final_mask > 0).astype(np.uint8)
#     labeled_mask, num_cells = skm.label(binary_mask, return_num=True)

#     # Filter small regions and create labeled images
#     cell_areas = {}  # Dictionary to store cell number and area
#     if min_area > 0:
#         props = skm.regionprops(labeled_mask)
#         filtered_mask = np.zeros_like(labeled_mask)
#         valid_count = 0
#         labeled_image = final_mask.copy()
#         colored_labeled_image = img.copy()
#         for prop in props:
#             if prop.area >= min_area:
#                 filtered_mask[labeled_mask == prop.label] = 1
#                 valid_count += 1
#                 centroid = prop.centroid

#                 # Save cell area with its number
#                 cell_areas[valid_count] = prop.area

#                 # Dynamic font scale based on cell area
#                 font_scale = max(0.1, min(0.5, np.sqrt(prop.area) / 50))
#                 font_thickness = 1 if prop.area > 200 else 0

#                 # Draw number on the binary mask
#                 cv2.putText(
#                     labeled_image,
#                     str(valid_count),
#                     (int(centroid[1]), int(centroid[0])),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     font_scale,
#                     0,
#                     font_thickness
#                 )

#                 # Draw number on the colored image with a background rectangle
#                 text = str(valid_count)
#                 x, y = int(centroid[1]), int(centroid[0])
#                 padding = 4

#                 # Iteratively adjust font size and position to fit within image bounds
#                 max_iterations = 10
#                 iteration = 0
#                 while iteration < max_iterations:
#                     text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
#                     text_w, text_h = text_size
#                     rect_top_left = (x - padding, y - text_h - padding)
#                     rect_bottom_right = (x + text_w + padding, y + padding)

#                     fits = True
#                     if rect_top_left[0] < 0:
#                         x -= rect_top_left[0]
#                         fits = False
#                     if rect_bottom_right[0] > img_width:
#                         x -= (rect_bottom_right[0] - img_width)
#                         fits = False
#                     if rect_top_left[1] < 0:
#                         y -= rect_top_left[1]
#                         fits = False
#                     if rect_bottom_right[1] > img_height:
#                         y -= (rect_bottom_right[1] - img_height)
#                         fits = False

#                     if fits:
#                         break

#                     font_scale *= 0.8
#                     if font_scale < 0.05:
#                         font_scale = 0.05
#                         break
#                     iteration += 1

#                 # Final position and size after adjustments
#                 text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
#                 text_w, text_h = text_size
#                 rect_top_left = (x - padding, y - text_h - padding)
#                 rect_bottom_right = (x + text_w + padding, y + padding)

#                 cv2.rectangle(
#                     colored_labeled_image,
#                     rect_top_left,
#                     rect_bottom_right,
#                     (0, 0, 0),
#                     -1
#                 )

#                 cv2.putText(
#                     colored_labeled_image,
#                     text,
#                     (x, y),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     font_scale,
#                     (255, 255, 255),
#                     font_thickness
#                 )
#         num_cells = valid_count
#         final_mask = filtered_mask * 255

#     return final_mask, num_cells, labeled_image, colored_labeled_image, cell_areas

# if __name__ == '__main__':
#     import argparse, os

#     parser = argparse.ArgumentParser(description='Segment image and count white cells')
#     parser.add_argument('-m', '--model', default='./final_semseg_coniferen_model.pth',
#                         help='model with pretrained weights')
#     parser.add_argument('-i', '--input', default='./input', help='path to the input file')
#     parser.add_argument('-o', '--output', default='./output/output.png', help='path where the output file should be stored')
#     args = parser.parse_args()
#     assert os.path.exists(args.model), 'model path does not exist'
#     assert os.path.exists(args.input), 'input path does not exist'

#     ### Normalize output path
#     if not args.output.endswith('.png'):
#         args.output = args.output + '.png'

#     ### Prepare Model ###
#     print('load model')
#     model = SegModel()
#     model.load_state_dict(torch.load(args.model)['state_dict'])
#     model.eval()

#     ### Process file ###
#     print('run model')
#     prediction, cell_count, labeled_image, colored_labeled_image, cell_areas = run_model(args.input, model, min_area=50)
#     print(f'Number of white cells detected: {cell_count}')

#     ### Save output ###
#     print('save output')
#     os.makedirs(os.path.dirname(args.output), exist_ok=True)
#     # Save the original binary mask
#     success = cv2.imwrite(args.output, prediction)
#     if not success:
#         print(f"Failed to save binary mask to {args.output}")
#     else:
#         print(f"Binary mask saved as: {args.output}")
#     # Save the labeled binary image with numbers
#     labeled_output = args.output.replace('.png', '_labeled.png')
#     success = cv2.imwrite(labeled_output, labeled_image)
#     if not success:
#         print(f"Failed to save labeled binary image to {labeled_output}")
#     else:
#         print(f"Labeled binary image saved as: {labeled_output}")
#     # Save the labeled colored image with numbers
#     print(f"colored_labeled_image shape: {colored_labeled_image.shape}, dtype: {colored_labeled_image.dtype}")
#     colored_labeled_image = colored_labeled_image.astype(np.uint8)
#     colored_labeled_output = args.output.replace('.png', '_colored_labeled.png')
#     success = cv2.imwrite(colored_labeled_output, colored_labeled_image)
#     if not success:
#         print(f"Failed to save colored labeled image to {colored_labeled_output}")
#     else:
#         print(f"Labeled colored image saved as: {colored_labeled_output}")
#     print('finished successfully')

#     ### Calculate area for a specific cell ###
#     print(f"\nTotal number of cells: {cell_count}")
#     while True:
#         try:
#             cell_number = int(input("Enter a cell number to calculate its area (or enter 0 to exit): "))
#             if cell_number == 0:
#                 break
#             if cell_number < 1 or cell_number > cell_count:
#                 print(f"Error: Cell number must be between 1 and {cell_count}.")
#                 continue
#             area = cell_areas.get(cell_number)
#             if area is not None:
#                 print(f"Area of cell {cell_number}: {area} pixels")
#             else:
#                 print(f"Error: Cell number {cell_number} not found.")
#         except ValueError:
#             print("Error: Please enter a valid integer.")

#粗略计算细胞壁厚度
# import numpy as np
# import os, glob, cv2, torch
# import pytorch_lightning as pl
# import segmentation_models_pytorch as smp
# import torchvision.transforms as transforms
# import skimage.measure as skm
# from tqdm import tqdm

# class SegModel(pl.LightningModule):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.net = smp.Unet(encoder_weights=None, classes=2, encoder_name='resnet50')

#     def forward(self, x):
#         out = self.net(x)
#         out = torch.argmax(out, dim=1, keepdim=True).long()
#         return out

#     def predict(self, path):
#         img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
#         preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet50')
#         img = transforms.ToTensor()(preprocessing_fn(img)).float()
#         img = torch.unsqueeze(img, 0)
#         r = self(img)
#         processed = torch.argmax(torch.squeeze(r.detach()), dim=0).long().numpy()
#         labels = skm.label(processed).transpose((1, 0))
#         return labels

# def run_model(img_path, model, min_area=50):
#     """
#     Runs the model on patches of the image, concats the predictions together,
#     counts the number of white cells, labels each cell with a number, and returns cell information.
#     :param img_path: path to the image
#     :param model: segmentation model with the forward path specified
#     :param min_area: minimum area threshold to filter small noise regions
#     :return: tuple (binary mask, number of white cells, labeled image, cell info, binary mask, labeled mask, img)
#     """
#     img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
#     img_height, img_width = img.shape[:2]
#     img_pad = np.zeros((img.shape[0] + 224, img.shape[1] + 224, img.shape[2]), dtype=np.uint8)
#     label_pad = np.zeros((img.shape[0] + 224, img.shape[1] + 224), dtype=np.uint8)
#     img_pad[112:-112, 112:-112] = img
#     preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet50')

#     for i in range(img_pad.shape[0] // 800 + 1):
#         for j in range(img_pad.shape[1] // 800 + 1):
#             x, x1, y, y1 = 800 * i, 800 * i + 1024, 800 * j, 800 * j + 1024
#             to_much_x = x1 - img_pad.shape[0]
#             to_much_y = y1 - img_pad.shape[1]
#             if to_much_x > 0:
#                 x = img_pad.shape[0] - 1024
#                 x1 = img_pad.shape[0]
#             if to_much_y > 0:
#                 y = img_pad.shape[1] - 1024
#                 y1 = img_pad.shape[1]
#             input_img = img_pad[x:x1, y:y1].copy()
#             img1 = preprocessing_fn(input_img)
#             img1 = transforms.ToTensor()(img1).float()
#             img1 = torch.unsqueeze(img1, 0)
#             r = model.forward(img1)
#             processed = torch.squeeze(r.detach()).long().numpy()
#             labels = skm.label(processed).transpose((1, 0))
#             result_semseg = (labels.T > 0).astype(np.uint8) * 255
#             label_pad[x + 112:x1 - 112, y + 112:y1 - 112] = result_semseg[112:-112, 112:-112]

#     # Crop back to original size
#     final_mask = label_pad[112:-112, 112:-112]

#     # Count white cells (connected components)
#     binary_mask = (final_mask > 0).astype(np.uint8)
#     labeled_mask, num_cells = skm.label(binary_mask, return_num=True)

#     # Filter small regions and create a labeled image with numbers
#     cell_info = {}
#     if min_area > 0:
#         props = skm.regionprops(labeled_mask)
#         filtered_mask = np.zeros_like(labeled_mask, dtype=np.uint8)
#         valid_count = 0
#         labeled_image = final_mask.copy()
#         for prop in props:
#             if prop.area >= min_area:
#                 filtered_mask[labeled_mask == prop.label] = 1
#                 valid_count += 1
#                 centroid = prop.centroid
#                 cell_info[valid_count] = {
#                     'area': prop.area,
#                     'centroid': (int(centroid[0]), int(centroid[1]))
#                 }
#                 cv2.putText(
#                     labeled_image,
#                     str(valid_count),
#                     (int(centroid[1]), int(centroid[0])),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.5,
#                     0,
#                     1
#                 )
#         num_cells = valid_count
#         final_mask = (filtered_mask * 255).astype(np.uint8)

#     return final_mask, num_cells, labeled_image, cell_info, binary_mask, labeled_mask, img

# def calculate_cell_wall_thickness(binary_mask, labeled_mask, cell_number, cell_info):
#     """
#     Calculate the thickness of the cell wall above and below the specified cell, and return the coordinates of counted black pixels.
#     Thickness is defined as the number of black pixels divided by 2.
#     :param binary_mask: Binary mask (1 for white cells, 0 for cell walls)
#     :param labeled_mask: Labeled mask with cell numbers
#     :param cell_number: The number of the cell to analyze
#     :param cell_info: Dictionary containing cell information (area, centroid)
#     :return: tuple (upper thickness, lower thickness, upper black pixels, lower black pixels)
#     """
#     if cell_number not in cell_info:
#         return None, None, [], []

#     # Get the centroid of the cell
#     centroid_y, centroid_x = cell_info[cell_number]['centroid']
#     img_height, img_width = binary_mask.shape

#     # Ensure the starting position is within the cell (white pixel)
#     if binary_mask[centroid_y, centroid_x] != 1:
#         cell_mask = (labeled_mask == cell_number).astype(np.uint8)
#         coords = np.argwhere(cell_mask)
#         if len(coords) == 0:
#             return None, None, [], []
#         centroid_y, centroid_x = coords[np.argmin(np.sum((coords - [centroid_y, centroid_x])**2, axis=1))]

#     # Calculate upper cell wall thickness and record black pixel coordinates
#     upper_black_count = 0
#     upper_black_pixels = []
#     y = centroid_y
#     in_wall = False
#     while y >= 0:
#         y -= 1
#         if y < 0:
#             break
#         pixel_value = binary_mask[y, centroid_x]
#         if pixel_value == 0 and not in_wall:
#             in_wall = True
#         if in_wall and pixel_value == 0:
#             upper_black_count += 1
#             upper_black_pixels.append((y, centroid_x))
#         if in_wall and pixel_value == 1:
#             break
#     upper_thickness = upper_black_count / 2.0  # Thickness is black pixel count divided by 2

#     # Calculate lower cell wall thickness and record black pixel coordinates
#     lower_black_count = 0
#     lower_black_pixels = []
#     y = centroid_y
#     in_wall = False
#     while y < img_height:
#         y += 1
#         if y >= img_height:
#             break
#         pixel_value = binary_mask[y, centroid_x]
#         if pixel_value == 0 and not in_wall:
#             in_wall = True
#         if in_wall and pixel_value == 0:
#             lower_black_count += 1
#             lower_black_pixels.append((y, centroid_x))
#         if in_wall and pixel_value == 1:
#             break
#     lower_thickness = lower_black_count / 2.0  # Thickness is black pixel count divided by 2

#     return upper_thickness, lower_thickness, upper_black_pixels, lower_black_pixels

# if __name__ == '__main__':
#     import argparse, os

#     parser = argparse.ArgumentParser(description='Segment image and count white cells')
#     parser.add_argument('-m', '--model', default='./final_semseg_coniferen_model.pth',
#                         help='model with pretrained weights')
#     parser.add_argument('-i', '--input', default='./input', help='path to the input file')
#     parser.add_argument('-o', '--output', default='./output/output.png', help='path where the output file should be stored')
#     args = parser.parse_args()
#     assert os.path.exists(args.model), 'model path does not exist'
#     assert os.path.exists(args.input), 'input path does not exist'

#     ### Normalize output path
#     if not args.output.endswith('.png'):
#         args.output = args.output + '.png'

#     ### Prepare Model ###
#     print('load model')
#     model = SegModel()
#     model.load_state_dict(torch.load(args.model)['state_dict'])
#     model.eval()

#     ### Process file ###
#     print('run model')
#     prediction, cell_count, labeled_image, cell_info, binary_mask, labeled_mask, img = run_model(args.input, model, min_area=50)
#     print(f'Number of white cells detected: {cell_count}')

#     ### Save output ###
#     print('save output')
#     os.makedirs(os.path.dirname(args.output), exist_ok=True)
#     # Save the original binary mask
#     cv2.imwrite(args.output, prediction)
#     # Save the labeled image with numbers
#     labeled_output = args.output.replace('.png', '_labeled.png')
#     cv2.imwrite(labeled_output, labeled_image)
#     print(f"Labeled image saved as: {labeled_output}")
#     print('finished successfully')

#     ### Calculate area and cell wall thickness for a specific cell ###
#     print(f"\nTotal number of cells: {cell_count}")
#     while True:
#         try:
#             cell_number = int(input("Enter a cell number to calculate its area and cell wall thickness (or enter 0 to exit): "))
#             if cell_number == 0:
#                 break
#             if cell_number < 1 or cell_number > cell_count:
#                 print(f"Error: Cell number must be between 1 and {cell_count}.")
#                 continue
#             if cell_number in cell_info:
#                 # Calculate area
#                 area = cell_info[cell_number]['area']
#                 print(f"Area of cell {cell_number}: {area} pixels")
                
#                 # Calculate cell wall thickness and get black pixel coordinates
#                 upper_thickness, lower_thickness, upper_black_pixels, lower_black_pixels = calculate_cell_wall_thickness(
#                     binary_mask, labeled_mask, cell_number, cell_info
#                 )
#                 if upper_thickness is not None and lower_thickness is not None:
#                     print(f"Upper cell wall thickness of cell {cell_number}: {upper_thickness:.2f} pixels")
#                     print(f"Lower cell wall thickness of cell {cell_number}: {lower_thickness:.2f} pixels")

#                     # Visualize the counted black pixels with thicker lines
#                     img_copy = img.copy()  # Use the original color image
#                     img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
#                     # Overlay the labeled image for better context
#                     labeled_overlay = cv2.cvtColor(labeled_image, cv2.COLOR_GRAY2BGR)
#                     img_copy = cv2.addWeighted(img_copy, 0.7, labeled_overlay, 0.3, 0)

#                     # Mark the centroid
#                     centroid_y, centroid_x = cell_info[cell_number]['centroid']
#                     cv2.circle(img_copy, (centroid_x, centroid_y), 5, (255, 0, 0), -1)  # Blue dot for centroid

#                     # Draw upper black pixels as a thick red line
#                     if upper_black_pixels:
#                         for i in range(len(upper_black_pixels) - 1):
#                             y1, x1 = upper_black_pixels[i]
#                             y2, x2 = upper_black_pixels[i + 1]
#                             cv2.line(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red line, thickness 3

#                     # Draw lower black pixels as a thick green line
#                     if lower_black_pixels:
#                         for i in range(len(lower_black_pixels) - 1):
#                             y1, x1 = lower_black_pixels[i]
#                             y2, x2 = lower_black_pixels[i + 1]
#                             cv2.line(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green line, thickness 3

#                     # Save the visualization
#                     vis_output = args.output.replace('.png', f'_cell_{cell_number}_thickness_vis.png')
#                     cv2.imwrite(vis_output, img_copy)
#                     print(f"Cell wall thickness visualization saved as: {vis_output}")
#                 else:
#                     print(f"Error: Unable to calculate cell wall thickness for cell {cell_number}.")
#             else:
#                 print(f"Error: Cell number {cell_number} not found.")
#         except ValueError:
#             print("Error: Please enter a valid integer.")


#标记年轮以及计算每一个区域的细胞总数
import numpy as np
import os, glob, cv2, torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms
import skimage.measure as skm
from tqdm import tqdm
from sklearn.cluster import KMeans

class SegModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = smp.Unet(encoder_weights=None, classes=2, encoder_name='resnet50')

    def forward(self, x):
        out = self.net(x)
        out = torch.argmax(out, dim=1, keepdim=True).long()
        return out

    def predict(self, path):
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet50')
        img = transforms.ToTensor()(preprocessing_fn(img)).float()
        img = torch.unsqueeze(img, 0)
        r = self(img)
        processed = torch.argmax(torch.squeeze(r.detach()), dim=0).long().numpy()
        labels = skm.label(processed).transpose((1, 0))
        return labels

def run_model(img_path, model, min_area=50):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img_height, img_width = img.shape[:2]
    img_pad = np.zeros((img.shape[0] + 224, img.shape[1] + 224, img.shape[2]), dtype=np.uint8)
    label_pad = np.zeros((img.shape[0] + 224, img.shape[1] + 224), dtype=np.uint8)
    img_pad[112:-112, 112:-112] = img
    preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet50')

    for i in range(img_pad.shape[0] // 800 + 1):
        for j in range(img_pad.shape[1] // 800 + 1):
            x, x1, y, y1 = 800 * i, 800 * i + 1024, 800 * j, 800 * j + 1024
            to_much_x = x1 - img_pad.shape[0]
            to_much_y = y1 - img_pad.shape[1]
            if to_much_x > 0:
                x = img_pad.shape[0] - 1024
                x1 = img_pad.shape[0]
            if to_much_y > 0:
                y = img_pad.shape[1] - 1024
                y1 = img_pad.shape[1]
            input_img = img_pad[x:x1, y:y1].copy()
            img1 = preprocessing_fn(input_img)
            img1 = transforms.ToTensor()(img1).float()
            img1 = torch.unsqueeze(img1, 0)
            r = model.forward(img1)
            processed = torch.squeeze(r.detach()).long().numpy()
            labels = skm.label(processed).transpose((1, 0))
            result_semseg = (labels.T > 0).astype(np.uint8) * 255
            label_pad[x + 112:x1 - 112, y + 112:y1 - 112] = result_semseg[112:-112, 112:-112]

    final_mask = label_pad[112:-112, 112:-112]
    binary_mask = (final_mask > 0).astype(np.uint8)
    labeled_mask, num_cells = skm.label(binary_mask, return_num=True)

    cell_info = {}
    if min_area > 0:
        props = skm.regionprops(labeled_mask)
        filtered_mask = np.zeros_like(labeled_mask, dtype=np.uint8)
        valid_count = 0
        labeled_image = final_mask.copy()
        for prop in props:
            if prop.area >= min_area:
                filtered_mask[labeled_mask == prop.label] = 1
                valid_count += 1
                centroid = prop.centroid
                cell_info[valid_count] = {
                    'area': prop.area,
                    'centroid': (int(centroid[0]), int(centroid[1])),
                    'label': prop.label
                }
                # 调整文字位置和大小
                centroid_y, centroid_x = int(centroid[0]), int(centroid[1])
                text = str(valid_count)
                # 默认字体大小
                font_scale = 0.5
                thickness = 1
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

                # 初始文字位置
                text_x = centroid_x - text_width // 2
                text_y = centroid_y + text_height // 2

                # 检查文字是否超出图像边界
                is_out_of_bounds = (
                    text_x < 0 or
                    text_x + text_width > img_width or
                    text_y - text_height < 0 or
                    text_y > img_height
                )

                # 特别检查右边边界：增加余量
                is_near_right_edge = centroid_x > img_width - 20  # 更宽松的右边边界检测

                if is_out_of_bounds or is_near_right_edge:
                    # 优先调整位置
                    if text_x < 0:
                        text_x = 0
                    if text_x + text_width > img_width:
                        text_x = img_width - text_width - 10  # 增加右边偏移余量
                    if text_y - text_height < 0:
                        text_y = text_height
                    if text_y > img_height:
                        text_y = img_height

                    # 重新检查调整后的位置是否仍然超出边界
                    is_still_out_of_bounds = (
                        text_x + text_width > img_width or
                        text_y > img_height
                    )

                    if is_still_out_of_bounds or is_near_right_edge:
                        # 如果调整位置后仍然超出，减小字体大小
                        font_scale = 0.3
                        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                        # 重新计算位置
                        text_x = centroid_x - text_width // 2
                        text_y = centroid_y + text_height // 2
                        # 确保不超出边界
                        text_x = max(0, min(text_x, img_width - text_width - 10))
                        text_y = max(text_height, min(text_y, img_height))

                cv2.putText(
                    labeled_image,
                    text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    0,  # 黑色文字
                    thickness
                )
        num_cells = valid_count
        final_mask = (filtered_mask * 255).astype(np.uint8)

    return final_mask, num_cells, labeled_image, cell_info, binary_mask, labeled_mask, img

def calculate_cell_wall_thickness(binary_mask, labeled_mask, cell_number, cell_info):
    if cell_number not in cell_info:
        return None, None, [], [], None

    centroid_y, centroid_x = cell_info[cell_number]['centroid']
    img_height, img_width = binary_mask.shape

    if binary_mask[centroid_y, centroid_x] != 1:
        cell_mask = (labeled_mask == cell_number).astype(np.uint8)
        coords = np.argwhere(cell_mask)
        if len(coords) == 0:
            return None, None, [], [], None
        centroid_y, centroid_x = coords[np.argmin(np.sum((coords - [centroid_y, centroid_x])**2, axis=1))]

    # 计算上细胞壁厚度
    upper_black_count = 0
    upper_black_pixels = []
    y = centroid_y
    in_wall = False
    while y >= 0:
        y -= 1
        if y < 0:
            break
        pixel_value = binary_mask[y, centroid_x]
        if pixel_value == 0 and not in_wall:
            in_wall = True
        if in_wall and pixel_value == 0:
            upper_black_count += 1
            upper_black_pixels.append((y, centroid_x))
        if in_wall and pixel_value == 1:
            break
    upper_thickness = upper_black_count

    # 计算下细胞壁厚度
    lower_black_count = 0
    lower_black_pixels = []
    y = centroid_y
    in_wall = False
    while y < img_height:
        y += 1
        if y >= img_height:
            break
        pixel_value = binary_mask[y, centroid_x]
        if pixel_value == 0 and not in_wall:
            in_wall = True
        if in_wall and pixel_value == 0:
            lower_black_count += 1
            lower_black_pixels.append((y, centroid_x))
        if in_wall and pixel_value == 1:
            break
    lower_thickness = lower_black_count

    # 计算直径：从质心向上和向下遍历白色像素点（binary_mask == 1）的总数
    upper_white_count = 0
    y = centroid_y
    while y >= 0:
        y -= 1
        if y < 0:
            break
        if binary_mask[y, centroid_x] == 1:
            upper_white_count += 1
        else:
            break

    lower_white_count = 0
    y = centroid_y
    while y < img_height:
        y += 1
        if y >= img_height:
            break
        if binary_mask[y, centroid_x] == 1:
            lower_white_count += 1
        else:
            break

    # 直径 = 向上白色像素数 + 向下白色像素数 + 1（包括质心本身）
    diameter = upper_white_count + lower_white_count + 1

    return upper_thickness, lower_thickness, upper_black_pixels, lower_black_pixels, diameter

def mark_cells_in_logical_rows(img, labeled_mask, cell_info, logical_rows_to_mark):
    """
    Mark all cells in the specified logical rows with blue contours.
    :param img: Original image (BGR)
    :param labeled_mask: Labeled mask
    :param cell_info: Dictionary containing cell information
    :param logical_rows_to_mark: List of logical row indices to mark
    :return: Image with marked cells
    """
    img_copy = img.copy()
    if not isinstance(img_copy, np.ndarray) or img_copy.shape[2] != 3:
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)

    cells_to_mark = [key for key, info in cell_info.items() if info['logical_row'] in logical_rows_to_mark]
    print(f"Marking {len(cells_to_mark)} cells in logical rows {logical_rows_to_mark} with blue contours")

    for key in cells_to_mark:
        label = cell_info[key]['label']
        cell_mask = (labeled_mask == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #thickness = 6 if key in smallest_cell_keys else 2
        cv2.drawContours(img_copy, contours, -1, (255, 0, 0), 6)  # Blue contours

    return img_copy

def find_and_mark_logical_rows(img, labeled_mask, cell_info, smallest_cell_keys, img_height, img_width, num_logical_rows=50, min_row_distance=5):
    """
    Find consecutive two logical rows where small cells make up at least 75% of the total cells,
    mark all cells in those rows with blue contours, and calculate the number of cells in each region
    excluding the marked rows.
    :param img: Original image (BGR)
    :param labeled_mask: Labeled mask
    :param cell_info: Dictionary containing cell information
    :param smallest_cell_keys: Keys of the smallest cells
    :param img_height: Height of the image
    :param img_width: Width of the image
    :param num_logical_rows: Number of physical rows to cluster into
    :param min_row_distance: Minimum logical row distance between the two boundaries
    :return: Image with marked cells, list of regions with cell counts
    """
    # Step 1: Extract centroid_y for all cells and cluster them into physical rows
    centroid_ys = np.array([info['centroid'][0] for info in cell_info.values()])
    if len(centroid_ys) < num_logical_rows:
        num_logical_rows = len(centroid_ys)  # Adjust if fewer cells than expected rows

    # Use K-means to cluster cells into physical rows based on centroid_y
    kmeans = KMeans(n_clusters=num_logical_rows, random_state=0).fit(centroid_ys.reshape(-1, 1))
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_.flatten()

    # Sort cluster centers to ensure logical rows are ordered from top to bottom
    sorted_indices = np.argsort(cluster_centers)
    cluster_centers = cluster_centers[sorted_indices]
    # Reassign labels based on sorted cluster centers
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_indices)}
    logical_rows = np.array([label_mapping[label] for label in labels])

    # Assign logical row to each cell in cell_info (ensuring uniqueness)
    for idx, (key, info) in enumerate(cell_info.items()):
        info['logical_row'] = logical_rows[idx]

    # Verify uniqueness: each cell should belong to exactly one logical row
    logical_row_assignments = {}
    for key, info in cell_info.items():
        logical_row = info['logical_row']
        if key in logical_row_assignments:
            raise ValueError(f"Cell {key} is assigned to multiple logical rows: {logical_row_assignments[key]} and {logical_row}")
        logical_row_assignments[key] = logical_row

    #print(f"Number of logical rows identified: {num_logical_rows}")
    #print(f"Cells per logical row (average): {len(cell_info) / num_logical_rows:.2f}")

    # Step 2: Count the number of small cells in each logical row
    row_counts = [0] * num_logical_rows  # Small cells per row
    row_total_counts = [0] * num_logical_rows  # Total cells per row
    for key in cell_info:
        logical_row = cell_info[key]['logical_row']
        row_total_counts[logical_row] += 1
        if key in smallest_cell_keys:
            row_counts[logical_row] += 1

    # Print the number of cells and small cells in each row for debugging
    for i in range(num_logical_rows):
        if row_total_counts[i] > 0:
            small_cell_ratio = row_counts[i] / row_total_counts[i]
            #print(f"Logical Row {i}: {row_total_counts[i]} cells, {row_counts[i]} small cells, small cell ratio: {small_cell_ratio:.2f}")
        else:
            print(f"Logical Row {i}: 0 cells, 0 small cells, small cell ratio: N/A")

    # Step 3: Find consecutive two logical rows where small cells make up at least 75% of the total
    row_pairs = []
    for logical_row in range(num_logical_rows - 1):
        total_cells = row_total_counts[logical_row] + row_total_counts[logical_row + 1]
        small_cells = row_counts[logical_row] + row_counts[logical_row + 1]
        if total_cells > 0:
            small_cell_ratio = small_cells / total_cells
            if small_cell_ratio >= 0.75:
                row_pairs.append((logical_row, logical_row + 1, small_cells, small_cell_ratio))
                #print(f"Rows ({logical_row}, {logical_row + 1}): {small_cells} small cells, ratio: {small_cell_ratio:.2f}")

    # Sort by the number of small cells in descending order
    row_pairs.sort(key=lambda x: x[2], reverse=True)

    # Step 4: Select non-overlapping pairs
    selected_pairs = []
    for pair in row_pairs:
        row1, row2, small_cells, ratio = pair
        # Check for overlap with already selected pairs
        overlap = False
        for selected_row1, selected_row2, _, _ in selected_pairs:
            if abs(row1 - selected_row1) < min_row_distance and abs(row2 - selected_row2) < min_row_distance:
                overlap = True
                break
        if not overlap:
            selected_pairs.append((row1, row2, small_cells, ratio))

    # Step 5: Mark all cells in the selected logical rows with blue contours
    img_copy = img.copy()
    marked_rows = set()
    if not selected_pairs:
        print("No consecutive two logical rows found with small cell ratio >= 75%. No cells will be marked.")
    else:
        #print(f"Found {len(selected_pairs)} pairs of consecutive logical rows with small cell ratio >= 75%.")
        for i, (logical_row1, logical_row2, small_cells, ratio) in enumerate(selected_pairs):
            # Get the logical rows to mark
            rows_to_mark = [logical_row1, logical_row2]
            marked_rows.update(rows_to_mark)
            #print(f"Marking cells in Logical Rows ({logical_row1}, {logical_row2}) with {small_cells} small cells, small cell ratio: {ratio:.2f}")
            # Mark all cells in these logical rows
            img_copy = mark_cells_in_logical_rows(img_copy, labeled_mask, cell_info, rows_to_mark)

    # Step 6: Calculate the number of cells in each region (excluding marked rows)
    regions = []
    current_region_start = 0
    marked_rows = sorted(marked_rows)  # Sort marked rows for region splitting

    for marked_row in marked_rows + [num_logical_rows]:  # Add the end to handle the last region
        if current_region_start < marked_row:
            region_rows = list(range(current_region_start, marked_row))
            if region_rows:  # Only add non-empty regions
                region_cell_count = sum(row_total_counts[row] for row in region_rows)
                regions.append((region_rows, region_cell_count))
                print(f"Region from Logical Row {region_rows[0]} to {region_rows[-1]}: {region_cell_count} cells")
        current_region_start = marked_row + 1

    return img_copy, regions

def determine_small_cell_threshold(cell_info, num_logical_rows=50, target_ratio=0.75):
    """
    Determine the area threshold for small cells such that some consecutive two logical rows
    have a small cell ratio of at least 75%.
    :param cell_info: Dictionary containing cell information
    :param num_logical_rows: Number of logical rows
    :param target_ratio: Target small cell ratio (default 0.75)
    :return: Area threshold for small cells, list of small cell keys
    """
    # Step 1: Sort cells by area
    areas = [(key, info['area']) for key, info in cell_info.items()]
    areas.sort(key=lambda x: x[1])  # Sort by area (ascending)
    total_cells = len(areas)

    # Step 2: Cluster cells into logical rows based on centroid_y
    centroid_ys = np.array([info['centroid'][0] for info in cell_info.values()])
    if len(centroid_ys) < num_logical_rows:
        num_logical_rows = len(centroid_ys)
    kmeans = KMeans(n_clusters=num_logical_rows, random_state=0).fit(centroid_ys.reshape(-1, 1))
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_.flatten()
    sorted_indices = np.argsort(cluster_centers)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_indices)}
    logical_rows = np.array([label_mapping[label] for label in labels])

    for idx, (key, info) in enumerate(cell_info.items()):
        info['logical_row'] = logical_rows[idx]

    # Step 3: Start with a higher percentage of small cells (e.g., 40%)
    initial_percentage = 0.4
    initial_index = int(total_cells * initial_percentage)
    area_threshold = areas[initial_index][1]
    #print(f"Initial area threshold (top {initial_percentage*100}%): {area_threshold}")

    # Step 4: Iterate to find the threshold that satisfies the condition
    for i in range(initial_index, total_cells):
        area_threshold = areas[i][1]
        smallest_cell_keys = [key for key, area in areas[:i+1]]

        # Count small cells in each logical row
        row_counts = [0] * num_logical_rows
        row_total_counts = [0] * num_logical_rows
        for key in cell_info:
            logical_row = cell_info[key]['logical_row']
            row_total_counts[logical_row] += 1
            if key in smallest_cell_keys:
                row_counts[logical_row] += 1

        # Check if any consecutive two rows have a small cell ratio >= 75%
        found = False
        for logical_row in range(num_logical_rows - 1):
            total_cells = row_total_counts[logical_row] + row_total_counts[logical_row + 1]
            small_cells = row_counts[logical_row] + row_counts[logical_row + 1]
            if total_cells > 0:
                small_cell_ratio = small_cells / total_cells
                if small_cell_ratio >= target_ratio:
                    found = True
                    break
        if found:
            #print(f"Found area threshold: {area_threshold}, with {i+1} small cells")
            return area_threshold, smallest_cell_keys

    # If no threshold satisfies the condition, use the initial threshold
    print(f"No threshold found to satisfy {target_ratio*100}% small cell ratio. Using initial threshold: {area_threshold}")
    return area_threshold, [key for key, area in areas[:initial_index+1]]

if __name__ == '__main__':
    import argparse, os
    import pandas as pd  # 导入 pandas 库

    parser = argparse.ArgumentParser(description='Segment image and count white cells')
    parser.add_argument('-m', '--model', default='./final_semseg_coniferen_model.pth',
                        help='model with pretrained weights')
    parser.add_argument('-i', '--input', default='./input', help='path to the input file')
    parser.add_argument('-o', '--output', default='./output/output.png', help='path where the output file should be stored')
    args = parser.parse_args()
    assert os.path.exists(args.model), 'model path does not exist'
    assert os.path.exists(args.input), 'input path does not exist'

    ### Normalize output path
    if not args.output.endswith('.png'):
        args.output = args.output + '.png'

    ### Prepare Model ###
    print('load model')
    model = SegModel()
    model.load_state_dict(torch.load(args.model)['state_dict'])
    model.eval()

    ### Process file ###
    print('run model')
    prediction, cell_count, labeled_image, cell_info, binary_mask, labeled_mask, img = run_model(args.input, model, min_area=50)

    ### Save output ###
    print('save output')
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cv2.imwrite(args.output, prediction)  # 保存 002.png（黑白二值图像）
    labeled_output = args.output.replace('.png', '_labeled.png')
    cv2.imwrite(labeled_output, labeled_image)  # 保存 002_labeled.png
    print(f"Labeled image saved as: {labeled_output}")

    ### 在原始彩色图像上绘制细胞编号 ###
    colored_labeled_image = img.copy()  # 使用原始彩色图像
    colored_labeled_image = cv2.cvtColor(colored_labeled_image, cv2.COLOR_RGB2BGR)  # 转换为 BGR 格式以绘制
    img_height, img_width = img.shape[:2]
    for cell_number, info in cell_info.items():
        centroid_y, centroid_x = info['centroid']
        text = str(cell_number)
        # 默认字体大小
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        # 初始文字位置
        text_x = centroid_x - text_width // 2
        text_y = centroid_y + text_height // 2

        # 检查文字是否超出图像边界
        is_out_of_bounds = (
            text_x < 0 or
            text_x + text_width > img_width or
            text_y - text_height < 0 or
            text_y > img_height
        )

        # 特别检查右边边界：增加余量
        is_near_right_edge = centroid_x > img_width - 20  # 更宽松的右边边界检测

        if is_out_of_bounds or is_near_right_edge:
            # 优先调整位置
            if text_x < 0:
                text_x = 0
            if text_x + text_width > img_width:
                text_x = img_width - text_width - 10  # 增加右边偏移余量
            if text_y - text_height < 0:
                text_y = text_height
            if text_y > img_height:
                text_y = img_height

            # 重新检查调整后的位置是否仍然超出边界
            is_still_out_of_bounds = (
                text_x + text_width > img_width or
                text_y > img_height
            )

            if is_still_out_of_bounds or is_near_right_edge:
                # 如果调整位置后仍然超出，减小字体大小
                font_scale = 0.3
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                # 重新计算位置
                text_x = centroid_x - text_width // 2
                text_y = centroid_y + text_height // 2
                # 确保不超出边界
                text_x = max(0, min(text_x, img_width - text_width - 10))
                text_y = max(text_height, min(text_y, img_height))

        cv2.putText(
            colored_labeled_image,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),  # 黑色文字
            thickness
        )
    colored_labeled_output = args.output.replace('.png', '_colored_labeled.png')
    cv2.imwrite(colored_labeled_output, colored_labeled_image)
    print(f"Colored labeled image saved as: {colored_labeled_output}")

    print('finished successfully')

    ### Determine the area threshold for small cells ###
    print("\nDetermining small cell threshold...")
    num_logical_rows = max(10, min(50, len(cell_info) // 7))
    area_threshold, smallest_cell_keys = determine_small_cell_threshold(cell_info, num_logical_rows=num_logical_rows, target_ratio=0.75)

    ### Find the consecutive two logical rows with the most small cells (ratio >= 75%), mark all cells in those rows, and calculate region cell counts ###
    boundary_image, regions = find_and_mark_logical_rows(
        img, labeled_mask, cell_info, smallest_cell_keys, img.shape[0], img.shape[1], num_logical_rows=num_logical_rows, min_row_distance=5
    )
    boundary_output = args.output.replace('.png', '_marked_rows.png')
    cv2.imwrite(boundary_output, boundary_image)
    print(f"Image with marked rows saved as: {boundary_output}")

    ### Calculate area, cell wall thickness, and diameter for specific cells ###
    print(f"\nTotal number of cells: {cell_count}")

    # 初始化一张基于黑白二值图像的图像，用于叠加所有细胞壁的可视化
    combined_vis_image = prediction.copy()  # 使用 002.png（黑白二值图像）
    combined_vis_image = cv2.cvtColor(combined_vis_image, cv2.COLOR_GRAY2BGR)  # 转换为 BGR 格式以绘制彩色线条

    # 复制 prediction 用于叠加细胞壁可视化到 002.png 上
    binary_with_thickness = prediction.copy()
    binary_with_thickness = cv2.cvtColor(binary_with_thickness, cv2.COLOR_GRAY2BGR)

    # 初始化一个列表，用于存储计算结果
    results = []

    while True:
        try:
            cell_number = int(input("Enter a cell number to calculate its area, cell wall thickness, and diameter (or enter 0 to exit): "))
            if cell_number == 0:
                # 保存叠加了所有细胞壁可视化的黑白二值图像
                combined_output = args.output.replace('.png', '_combined_thickness_vis.png')
                cv2.imwrite(combined_output, combined_vis_image)
                print(f"Combined cell wall thickness visualization saved as: {combined_output}")

                # 将结果保存到 Excel 表格
                if results:  # 只有在有数据时才保存
                    df = pd.DataFrame(results, columns=[
                        '细胞编号', '细胞面积', '细胞直径', '上细胞壁厚度', '下细胞壁厚度'
                    ])
                    excel_output = args.output.replace('.png', '_cell_measurements.xlsx')
                    df.to_excel(excel_output, index=False)
                    print(f"Cell measurements saved to Excel as: {excel_output}")
                else:
                    print("No cell measurements to save.")
                break

            if cell_number < 1 or cell_number > cell_count:
                print(f"Error: Cell number must be between 1 and {cell_count}.")
                continue

            if cell_number in cell_info:
                area = cell_info[cell_number]['area']
                print(f"Area of cell {cell_number}: {area} pixels")

                upper_thickness, lower_thickness, upper_black_pixels, lower_black_pixels, diameter = calculate_cell_wall_thickness(
                    binary_mask, labeled_mask, cell_number, cell_info
                )
                if upper_thickness is not None and lower_thickness is not None:
                    print(f"Upper cell wall thickness of cell {cell_number}: {upper_thickness:.2f} pixels")
                    print(f"Lower cell wall thickness of cell {cell_number}: {lower_thickness:.2f} pixels")
                    print(f"Diameter of cell {cell_number}: {diameter} pixels")

                    # 将结果追加到列表中
                    results.append({
                        '细胞编号': cell_number,
                        '细胞面积': area,
                        '细胞直径': diameter,
                        '上细胞壁厚度': upper_thickness,
                        '下细胞壁厚度': lower_thickness
                    })

                    # 可视化细胞壁，叠加到 combined_vis_image 和 binary_with_thickness 上
                    centroid_y, centroid_x = cell_info[cell_number]['centroid']
                    cv2.circle(combined_vis_image, (centroid_x, centroid_y), 5, (255, 0, 0), -1)
                    cv2.circle(binary_with_thickness, (centroid_x, centroid_y), 5, (255, 0, 0), -1)

                    # 绘制上细胞壁（蓝色）
                    if upper_black_pixels:
                        for i in range(len(upper_black_pixels) - 1):
                            y1, x1 = upper_black_pixels[i]
                            y2, x2 = upper_black_pixels[i + 1]
                            cv2.line(combined_vis_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv2.line(binary_with_thickness, (x1, y1), (x2, y2), (0, 0, 255), 3)

                    # 绘制下细胞壁（绿色）
                    if lower_black_pixels:
                        for i in range(len(lower_black_pixels) - 1):
                            y1, x1 = lower_black_pixels[i]
                            y2, x2 = lower_black_pixels[i + 1]
                            cv2.line(combined_vis_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            cv2.line(binary_with_thickness, (x1, y1), (x2, y2), (0, 255, 0), 3)
                else:
                    print(f"Error: Unable to calculate cell wall thickness for cell {cell_number}.")
            else:
                print(f"Error: Cell number {cell_number} not found.")
        except ValueError:
            print("Error: Please enter a valid integer.")