#############################
#NVIDIA  All Rights Reserved
#Haoyu Yang 
#Design Automation Research
#Last Update: June 12 2024
#############################
from photonic_model import * 
import cv2
import numpy as np
#from datetime import datetime
if __name__=="__main__":

    id=0

    #solver = nvilt_engine_2(image_path='./benchmarks/photonics/stablering_2nm_8.png',morph = 0, scale_factor=5)
    # solver = nvilt_engine_2(image_path='./eps.png',morph = 0, scale_factor=5)
    solver = nvilt_engine_2(image_path='./eps.png',morph = 0, scale_factor=1)

    #print(nvilt1.target_image.shape, torch.max(nvilt1.target_image), torch.min(nvilt1.target_image))
    
    from datetime import datetime



    s=datetime.now()

    for iter in range(20):
        solver.optimize_s()
        print("%s: iter %g: loss: %.3f, loss_l2: %.3f, loss_pvb: %.3f + %.3f"%(datetime.now(), solver.iteration, solver.loss, solver.loss_l2, solver.loss_pvb_i, solver.loss_pvb_o))

    e=datetime.now()
    

    with torch.no_grad():
        #solver.nvilt.mask.data = nn.functional.interpolate(input=solver.nvilt.avepool_lres(solver.nvilt.mask_s).data, scale_factor=solver.nvilt.scale_factor, mode = 'bicubic', align_corners=False, antialias=True)

        if False:
        
            mask, cmask, x_out, x_out_max, x_out_min = solver.nvilt.forward_test()
            
            results = evaluation(mask, solver.target, x_out, x_out_min, x_out_max) 
            l2 = results.get_l2()
            pvb=results.get_pvb()
            #epe=results.get_epe()
            print("Design (morph): %g, runtime is: %s, final L2 is: %g, final PVB is: %g"%(id, e-s,l2, pvb))

            final_image = torch.cat((solver.target, mask, solver.nvilt.aerial, x_out), dim=3).cpu().detach().numpy()[0,0,:,:]*255
            #mask_image = mask.cpu().detach().numpy()[0,0,:,:]*255
        


            cv2.imwrite(solver.image_path+".final_morph.png", final_image)


        mask, x_out, x_out_max, x_out_min = solver.nvilt.forward_batch_test(use_morph=False)
        
        results = evaluation(mask, solver.target_s, x_out, x_out_min, x_out_max) 
        l2 = results.get_l2()
        pvb=results.get_pvb()
        print("Design (no morph): %g, runtime is: %s, final L2 is: %g, final PVB is: %g"%(id, e-s,l2, pvb))

        final_image = torch.cat((solver.target_s, mask, x_out), dim=3).cpu().detach().numpy()[0,0,:,:]*255
        #mask_image = mask.cpu().detach().numpy()[0,0,:,:]*255
    
        if True:
            mm = mask.cpu().detach().numpy()[0,0,:,:]*255
            print("this is the shape of the mask", mm.shape)
            zz = x_out.cpu().detach().numpy()[0,0,:,:]*255
            print("this is the shape of the resist", zz.shape)
            cv2.imwrite(solver.image_path+".mask.png", mm)
            cv2.imwrite(solver.image_path+".resist.png", zz)

        cv2.imwrite(solver.image_path+".final.png", final_image)
        


