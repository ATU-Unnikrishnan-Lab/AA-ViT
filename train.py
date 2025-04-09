import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
import numpy as np, h5py 
# from skimage.measure import compare_psnr as psnr
from skimage.metrics import peak_signal_noise_ratio as psnr

import os
import matplotlib.pyplot as plt

import os
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import os
import numpy as np

# def visualize_results(fake_B, real_B, diff_map, saliency, epoch, save_dir):
#     os.makedirs(save_dir, exist_ok=True)  # Ensure save directory exists

#     if saliency is None:
#         fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#         titles = ["Predicted T1ce", "Real T1ce", "Difference Map"]
#         images = [fake_B, real_B, diff_map]
#         save_path = os.path.join(save_dir, f"epoch_{epoch}.png")

#         for i, ax in enumerate(axes):
#             ax.imshow(images[i], cmap="gray")
#             ax.set_title(titles[i])
#             ax.axis("off")
#     else:
#         # Normalize saliency map to [0, 1]
#         saliency = np.squeeze(saliency)
#         saliency_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

#         fig, axes = plt.subplots(1, 4, figsize=(20, 5))
#         titles = ["Predicted T1ce", "Real T1ce", "Difference Map", "Saliency Overlay"]
#         images = [fake_B, real_B, diff_map]

#         # Display standard grayscale images
#         for i in range(3):
#             axes[i].imshow(images[i], cmap="gray")
#             axes[i].set_title(titles[i])
#             axes[i].axis("off")

#         # Overlay saliency on top of fake_B (predicted image)
#         axes[3].imshow(np.squeeze(fake_B), cmap="gray")
#         axes[3].imshow(saliency_norm, cmap="jet", alpha=0.5)
#         axes[3].set_title(titles[3])
#         axes[3].axis("off")

#         save_path = os.path.join(save_dir, f"epoch_{epoch}_saliency.png")

#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()




def print_log(logger,message):
    print(message, flush=True)
    if logger:
        logger.write(str(message) + '\n')
if __name__ == '__main__':
    opt = TrainOptions().parse()
    #Training data
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)
    ##logger ##
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    logger = open(os.path.join(save_dir, 'log.txt'), 'w+')
    print_log(logger,opt.name)
    logger.close()
    #validation data
    opt.phase='val'
    data_loader_val = CreateDataLoader(opt)
    dataset_val = data_loader_val.load_data()
    dataset_size_val = len(data_loader_val)
    print('#Validation images = %d' % dataset_size)
    if opt.model=='cycle_gan':
        L1_avg=np.zeros([2,opt.niter + opt.niter_decay,len(dataset_val)])      
        psnr_avg=np.zeros([2,opt.niter + opt.niter_decay,len(dataset_val)])            
    else:
        L1_avg=np.zeros([opt.niter + opt.niter_decay,len(dataset_val)])      
        psnr_avg=np.zeros([opt.niter + opt.niter_decay,len(dataset_val)])       
    model = create_model(opt)
    visualizer = Visualizer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        #Training step
        opt.phase='train'
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                if opt.dataset_mode=='aligned_mat':
                    temp_visuals=model.get_current_visuals()
                    visualizer.display_current_results(temp_visuals, epoch, save_result)
                elif  opt.dataset_mode=='unaligned_mat':   
                    temp_visuals=model.get_current_visuals()
                    temp_visuals['real_A']=temp_visuals['real_A'][:,:,0:3]
                    temp_visuals['real_B']=temp_visuals['real_B'][:,:,0:3]
                    temp_visuals['fake_A']=temp_visuals['fake_A'][:,:,0:3]
                    temp_visuals['fake_B']=temp_visuals['fake_B'][:,:,0:3]
                    temp_visuals['rec_A']=temp_visuals['rec_A'][:,:,0:3]
                    temp_visuals['rec_B']=temp_visuals['rec_B'][:,:,0:3]
                    if opt.lambda_identity>0:
                      temp_visuals['idt_A']=temp_visuals['idt_A'][:,:,0:3]
                      temp_visuals['idt_B']=temp_visuals['idt_B'][:,:,0:3]                    
                    visualizer.display_current_results(temp_visuals, epoch, save_result)                    
                else:
                    temp_visuals=model.get_current_visuals()
                    visualizer.display_current_results(temp_visuals, epoch, save_result)                    
                    

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')

            iter_data_time = time.time()
        #Validaiton step
        if epoch % opt.save_epoch_freq == 0:
            logger = open(os.path.join(save_dir, 'log.txt'), 'a')
            print(opt.dataset_mode)
            opt.phase='val'
            for i, data_val in enumerate(dataset_val):
#        		    
                model.set_input(data_val)
#        		    
                fake_B_np, real_B_np, diff_map, _ = model.test(compute_saliency=True)
#        		    
                fake_im=model.fake_B.cpu().data.numpy()
#        		    
                real_im=model.real_B.cpu().data.numpy() 
#        		    
                real_im=real_im*0.5+0.5
#        		    
                fake_im=fake_im*0.5+0.5
                if real_im.max() <= 0:
                    continue
                L1_avg[epoch-1,i]=abs(fake_im-real_im).mean()
                psnr_avg[epoch-1,i]=psnr(fake_im/fake_im.max(),real_im/real_im.max())
#               
#    
#                 
            l1_avg_loss = np.mean(L1_avg[epoch-1])
#                
            mean_psnr = np.mean(psnr_avg[epoch-1])
#                
            std_psnr = np.std(psnr_avg[epoch-1])
#                
            print_log(logger,'Epoch %3d   l1_avg_loss: %.5f   mean_psnr: %.3f  std_psnr:%.3f ' % \
            (epoch, l1_avg_loss, mean_psnr,std_psnr))
#                
            print_log(logger,'')
            logger.close()
    		   #
            print('saving the model at the end of epoch %d, iters %d' %(epoch, total_steps))
#        		    
            model.save('latest')
#        		   
            model.save(epoch)
            # visualize_results(fake_B_np[0], real_B_np[0], diff_map[0], _[0], epoch, save_dir)


        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
