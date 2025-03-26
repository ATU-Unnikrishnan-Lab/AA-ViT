import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from util import html

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    visualizer = Visualizer(opt)

    # Create results webpage
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    # Test loop
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        
        model.set_input(data)

        img_path = model.get_image_paths()
        print('%04d: process image... %s' % (i, img_path))  # Ensuring print order
        
        model.test()  # Run model inference

        if opt.dataset_mode == 'aligned_mat':
            visuals = model.get_current_visuals()
            img_path[0] = img_path[0] + str(i)
        elif opt.dataset_mode == 'unaligned_mat':   
            visuals = model.get_current_visuals()
            slice_select = [opt.input_nc // 2] * 3  # Ensure integer index
            visuals['real_A'] = visuals['real_A'][:, :, slice_select]
            visuals['real_B'] = visuals['real_B'][:, :, slice_select]
            visuals['fake_A'] = visuals['fake_A'][:, :, slice_select]
            visuals['fake_B'] = visuals['fake_B'][:, :, slice_select]
            visuals['rec_A'] = visuals['rec_A'][:, :, slice_select]
            visuals['rec_B'] = visuals['rec_B'][:, :, slice_select]
            img_path[0] = img_path[0] + str(i)
        else:
            visuals = model.get_current_visuals()

        visualizer.save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio)

    # Compute final PSNR & SSIM after all images are processed
    model.compute_final_metrics()
    
    webpage.save()
