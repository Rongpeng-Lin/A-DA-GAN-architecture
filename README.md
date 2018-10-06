# A-DA-GAN-architecture
## A basic architecture of "DA-GAN: Instance-level Image Translation by Deep Attention Generative Adversarial Networks"<br>
&#8195;This is a basic architecture implementation, and the structure of the article is outlined below:<br>&#8195;&#8195;1.&#8195;The image is encoded using an encoder (convolutional architecture).<br>
&#8195;&#8195;2.&#8195;Use another set of convolutions combined with a full join to generate an attention area (similar to the border of the target detection) and perform a masking operation on the original image.<br>
&#8195;&#8195;3.&#8195;3.The mask operation does not use the 01 mask, but instead uses sigmoid instead of direction propagation, making the change more 'soft'.<br>
## how to use<br>
&#8195;1.&#8195;There are some samples marked incorrectly in the svhn data set, first clean the sample:<br>
&#8195;&#8195;python&#8195;D:\SVHN_dataset\train\DAE_GAN\Data.py&#8195;--op_type="clear"&#8195;--im_dir="D:/SVHN_dataset/train/forcmd/"&#8195;--raw_dir="D:/SVHN_dataset/train/forcmd/"&#8195;--if_clip=False<br>
&#8195;2.&#8195;Create positive and negative sample data:<br>
&#8195;&#8195;python&#8195;D:\SVHN_dataset\train\DAE_GAN\Data.py&#8195;--op_type="create"&#8195;--im_dir="D:/SVHN_dataset/train/forcmd/"&#8195;--raw_dir="D:/SVHN_dataset/train/forcmd/"&#8195;--if_clip=False<br>
&#8195;3.&#8195;Perform training or loading:<br>
&#8195;&#8195;Train:<br>
&#8195;&#8195;&#8195;python&#8195;D:\SVHN_dataset\train\DAE_GAN\train.py&#8195;--is_train="train"&#8195;--im_size=64&#8195;--batch=2&#8195;--epoch=100&#8195;--hw_size=30&#8195;--k=2e5&#8195;--alpa=0.9&#8195;--beta=0.5&#8195;--im_dir="D:/SVHN_dataset/train/forcmd/"&#8195;--save_dir="D:/SVHN_dataset/train/forcmd/ckpt/"&#8195;--saveS_dir="D:/SVHN_dataset/train/forcmd/SampleS/"&#8195;--saveT_dir="D:/SVHN_dataset/train/forcmd/SampleT/"<br>
&#8195;&#8195;Test:<br>
&#8195;&#8195;&#8195;python&#8195;D:\SVHN_dataset\train\DAE_GAN\train.py&#8195;--is_train="test"&#8195;--load_dir="D:/SVHN_dataset/train/ckpt/"&#8195;--raw_im_dir="D:/SVHN_dataset/test"&#8195;--save_im_dir="D:/SVHN_dataset/test_save/"<br>
