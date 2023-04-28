# sd-meebs
Training Stable Diffusion to produce meebs

## What is a meeb?

![Meeb](./site/meeb.png)

This is a meeb. He likes to use https://www.beemit.com.au/.

The training images are taken from the https://www.beemit.com.au/ marketing materials and are copyright Digital Wallet Pty. Ltd.

## Why?
Because we can.

## Project Structure
* `./img_X.Y` - Training images. Multiple directories for if the images change over time.
Since the cycle-count is included in the folder naming, changes to the folder names are
important, so we can version them using this folder structure. Sort of a version control
within the version control as it's easier to just change paths than to switch the whole
repo to a new branch when you're experimenting.
* `./logs` - Tensorboard log files. You can use these to visualize the learning rate, loss
rate, etc.
* `./reg` - Regularization images. There's a lot, but these hopefully don't need to change
much.
* `./site` - Documentation and supporting files for the README.
* `./*.json` - Config files that hold the training parameters.

## Versions
* 1.0 - First run using a large data set of 150-ish images. Transparent backgrounds, no cropping.
Results weren't fantastic, but surprising nonetheless.
* 1.1 - Lots of improvements to the training images.
  * **Culled all the different color variants** - Many of the images were just recolors of the same pose/object. I left the color variety by cycling the colors through the images roughly evenly.
  * **Removed transparent backgrounds** - SD really doesn’t like transparency, and I think this might have been the cause of the weird pixelization around the eyes. I gave each image a solid white or black background with an even distribution across the set.
  * **Remove the hanging chad** - Made sure the number of images divided evenly by the batch size so there’s not a ‘half-batch’ at the end. I just deleted one of the images… easy.
  * **Removed the cyclopses** - A couple of the images had a single ‘cyclops’ eye instead of two eyes. Since there was only a handful I ditched them rather than trying to differentiate the two. While it’s more than possible to train both, I wanted to start simple and aim for good results to build off.
* 1.3 - Cut the network ranks way back to try and reduce output file size. Both ranks were reduced to `4` and didn't seem to
dramatically change the results. Output dropped from 100Mb to 20Mb though, so well worth the reduction.

## Prerequisites

About 8GB of VRAM. You can probably squeeze by on 6GB, but more is definitely better.
Some of my notes on performance and hardware: https://n15g.notion.site/Performance-543adfdf100e4c29836e0b7cd188fe05

### Model training
* [Kohya_ss](https://github.com/bmaltais/kohya_ss) - This is the GUI we'll be using to train
our model. It's a python web app and makes it much easier to write the absurdly long prompts
that you'll need to feed into the training process as well as supports saving and loading
config files.
* [LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) - Adds a couple of new algorithms for training Low-Rank Adaptations
Specifically, we're going to be training using the LyCORIS/LoHA algorithm as it does well for styles.

### Inference (generating images)
* [Automatic1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) - Python web interface for Stable Diffusion.
By far the easiest way to get started using SD. It comes with a bundled version of the engine, all you need
is to add a model.
* [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) Said model. This is
the 'memory' of the AI, and contains the network weights and everything that the engine will 'understand'
by default. It's our starting point for further training the network.
* [LyCORIS plugin](https://github.com/KohakuBlueleaf/a1111-sd-webui-lycoris) - Lets you use LyCORIS
models in Automatic1111.
[gui-user.bat](..%2Fkohya_ss%2Fgui-user.bat)



## Training a model

1. Install the [Kohya_ss](https://github.com/bmaltais/kohya_ss) GUI and all dependencies (Python, etc).
2. Install the LyCORIS scripts by following the instructions [here](https://github.com/KohakuBlueleaf/LyCORIS#for-kohya-script).
3. Run the Kohya GUI and connect to it at http://127.0.0.1:7860/.
   * I generally like to run Kohya and Automatic1111 at the same time, so I usually
   change the port that Kohya runs on by editing the `gui-user.bat` file as follows:
   ```
   @echo off
    REM Example of how to start the GUI with custom arguments. In this case how to auto launch the browser:
    REM call gui.bat --inbrowser
    REM
    REM You can add many arguments on the same line
    REM
    call gui.bat --server_port 17860
   ```
   If you're on mac/linux, modify the `.sh` or `.ps1` files appropriately.
4. Switch to the `Dreambooth LoRA` tab and load one of the config files from the repo
    root directory.
5. You might need to tweak the folder paths to be non-relative.
    ![folders.png](site%2Ffolders.png)
6. Hit `Train model`.
7. Check the `output` folder for the results of the model training.
   * The `samples` directory will contain example images generated along the process and
   is a good way of visually tracking the progress and looking for over-fitting or
   a lack of training if your parameters are too weak.
   * For more than a single epoch, the resulting outputs will have suffixes including the
   epoch count, such as `-000001`. You can use this to output checkpoints along the way
   and pick a result that is in the Goldilocks zone of not under or over-trained.
   * For some reason, when using regularization images it only outputs every second epoch...
   I think this is because with regularization images you need to double the number of steps...
   but it's still weird and confusing... I'd rather have the partial epochs than have to double
   everything.


## Training parameters
My latest notes on what all these mean are here: https://n15g.notion.site/LoRA-Training-with-Kohya-SS-2f99a372cb9a44ee8c7e6820bb6b192a

### **Folders Tab**
- **Image folder** - Location of your training images.
    - ***NOTE*** - This is the root of the images directory. Within that your images will need to be in a folder structure that tells the trainer what subject and class is contained in the set, and the number of cycles to complete.
    - This will look like this `img/{cycles}_{subject} {class}`... `50_beem person`, `15_apple food`, etc.
    - Still figuring out the best keywords for subject and class, and omitted class entirely for the 1.1 training.
    - Since you'll usually target around 1500 steps for training, and will rarely have 1500 training images, you use the cycle count to tell the trainer to cycle over the same images `x` times.
- **Regularization folder** - Location of your regularization images.
    - Really useful for keeping the model from overwriting the concept of a more general class of your training subject... for instance, preventing every `person` becoming a `meeb` when using the model. Also useful for training the AI about what it's looking at. I used the regularization images here to try and get the AI to recognize meebs as a type of person... something that the base SD model was not prepared to make the leap too.
    - I'm still experimenting with when and how to use reg images properly. Gathering my notes [here](https://www.notion.so/Regularization-Images-6c6ef52ebce8429b8117e5c880b31c60).
- **Output folder** - Where the stuff comes out. duh.
    - **Logging folder** - Where the tensorboard logs will spit out... if you put nothing it will spit files out into your drive root folder... so better to set it even if you don't want logs.
- **Model output name** - This is the filename that the new model will spit out as.
    - **Training comment** - Extra bit of free-text that will be copied into the generated metadata for the model. I usually put a copy of the number of images and cycle count here since it's not included in the settings json. `50c 36i` for 50 cycles of 36 images;

### Training Settings Tab
- **Lora Type** - The type of LoRA we want to produce. In this case we're training a LyCORIS/LoHA.
- **Training batch size** - How many images to load into VRAM at once. More means more VRAM used but can also dilute the training... Most people seem to stick to 2 at a time.
- **Epoch** - How many ‘epochs’ or duplicates of the training to complete before finishing. Useful because you can output a version of the model at each epoch to get results as the model learns… you can use this to cherry pick a version from the middle before it starts overfitting for instance.
- **Save every N epochs** - How often to save an epoch.
- **Caption extension** - What the extension of your caption files is `.txt` usually.
- **Mixed precision / Save precision** - Higher precision means more accuracy, but larger files and more VRAM needed to load the model. Generally `fp16` (half-precision) is fine.
- **Number of CPU threads per core** - Usually 2, but if you don’t have a hyperthreaded CPU… get one, Jesus.
- **Seed** - Sets the input seed for the process. If you use the same seed on two runs you will get the same result. Useful to use the same seed when you’re experimenting to avoid randomness polluting the results.
- **Learning rate** - Your main lever for how fast the model learns your new subject. Lower is less likely to over fit, but may take orders of magnitude more time to learn. Too much and you’ll overshoot the gradient decent and wind up [out in infinity](https://developers.google.com/machine-learning/crash-course/fitter/graph).
    - You can actually set the learning rate of the text encoder and Unet differently, and this is generally advised. If you do, then this value is ignored.
    - **Text Encoder learning rate** - Usually a good place to start is `5e-5` or `0.00005`.
    - **Unet learning rate** - Usually a good place to start is `1e-4` or `0.0001`.
- **LR Scheduler** - Defines how the learning rate adjusts over time.
    - **constant** - What it says on the tin.
    - **polynomial** - Linear reduction to zero.
    - **cosine**- Nice smooth progression to zero.
    - ***x*-with restarts** - Normally the value is changed over the entire training duration, but *************with restarts************* it will periodically (per epoch, or configured with another setting) reset back to the highest value.
- **LR warmup** - Ease into the learning rate over a certain percentage of the steps. Not sure how this helps, but might prevent the AI from jumping on wrong features early and baking them in before getting the general gist.
- **Optimizer** - AdamW8Bit is usually fine. There are some great new optimizers that can automatically adjust the learning rate though. Great info here: [https://rentry.org/59xed3#optimizers](https://rentry.org/59xed3#optimizers)
- **Network Rank/Alpha -** The "network rank" or "network dimensions," represented as the `network_dim` parameter, indicates how detailed your fine-tuning model can get. More dimensions mean more layers available to absorb the finer details of your training set. The alpha acts like a choke on the rank and prevents it from overfitting too fast. Higher values means more damping.
    - See [Training Settings](https://www.notion.so/Training-Settings-6b6008839ba4496b9e312a933a3162a4) for good starting values.
- **Convolution Rank/Alpha** - Very similar to the above network values, but only apply to the convolution ranks of the LoRA, and only available when training a LoCon or LoHA LoRA. These are `2^dim`, so lower values than normal are used… 4 = a normal rank of 16 for instance.
- **Max Resolution** - SD1.5 is trained on 512x512 images, so you want to at least start there. Higher training resolution can help the training pick up on smaller details, but acts as a dampner on learning rate. If you increase the resolution it will slow down the process for a start as it needs to operate on a higher pixel space, but also need to increase the total number of cycles or increase the learning rate to compensate for the lower overall learning speed.

### Advanced Config
- **No token padding** - Still don’t know what this does.
- **Gradient accumulation steps** - Instead of training on one or two images at a time, based on the batch size… this will concatenate `batch size x acc steps` images at a time and apply the learning on the entire set. Theoretically produces more accurate results, but requires you to increase the step count appropriately so that you still get the same amount of training. Questionable value overall as the amount of extra training time is significant.
- **Clip skip** - [https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/5674](https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/5674)
- **Noise offset** - Introduces noise into the average pixel values preventing the models from always settling on an average luminance of 0.5. Basically this means that models produced with this setting have a higher dynamic range in the output. `0.06` is a good general value.
- **Use xformers** - If you have an NVIDIA graphics card, this will dramatically improve your processing speed (2-3x) at the cost of a small amount of non-determinism in the finer details of images. Always worth it unless you’re writing a paper or something and need 100% consistent results.

### Steps
Want to aim for a total step count in the 1500 to 2000 range.
`total_steps = (num_images * repeats * max_train_epochs) / train_batch_size`


## General advice
### Tensorboard
At the bottom of the page is a button to start Tensorboard. This will bring up a webpage
that visualizes the learning rate and loss rate over time. This is super useful for
fine-tuning your training parameters as you can correlate the values with the sample images
being produced to find where good learning and loss rates coincide with good results and then
iterate the training based off those values.
![tensorboard.png](site%2Ftensorboard.png)


## Using the generated models
1. Download and set up [Automatic1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui).
2. Install the [LyCORIS plugin](https://github.com/KohakuBlueleaf/a1111-sd-webui-lycoris).
3. Download the SD-1.5 model from [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) 
    and place it in the `models\Stable-diffusion` folder of the Automatic1111 install directory.
   * The [1.5-pruned-emaonly.safetensors](https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.safetensors)
   version will do.
4. Fire up Automatic1111 http://127.0.0.1:7860/.
5. From the checkpoint list select the SD model.
    ![checkpoint.png](site%2Fcheckpoint.png)
6. Take your output LoRA file from the training earlier and drop it into the `models\LyCORIS` folder
    in the Automatic1111 install directory. There is a `LoRA` folder as well, but we want to use the
    folder specifically for LyCORIS variant LoRA.
7. Enter a prompt using the `<lyco:{filename}:{weight}>` syntax.
   * For example: `masterpiece, best quality, 8k, drawing of a man in the beem style holding a laptop, <lyco:beem_style_1.3-000003:1>`
8. Profit!
![results.png](site%2Fresults.png)