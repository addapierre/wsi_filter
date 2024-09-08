# Computer Vision Analysis of Brachial Plexus Whole Slide Image  

https://github.com/user-attachments/assets/becb123d-7698-4a25-9ed4-dc4534ddfbe1


### 1. Introduction  
Brachial plexus paralysis refers to a condition resulting from injury to the brachial plexus, a network of nerves that originates from the spinal cord in the neck and extends into the arm. This network is crucial for controlling movement and sensation in the shoulder, arm, and hand. When these nerves are damaged, it can lead to weakness, paralysis, or loss of sensation in the affected arm, a condition often referred to as brachial palsy.  

  
<p align="center"><img src="https://www.medecin-dommage-corporel.expert/wp-content/uploads/2015/02/plexus-brachial.jpg" alt="brachial_plexus" width="300"/><br><i>View of a brachial plexus.</i> 

  
Brachial plexus paralysis injuries occurs primarily through traumatic events and affects mainly motor vehicle accident victims, but also newborn babies injured during complicated deliveries, where the baby's shoulder may become stuck, leading to stretching of the brachial plexus nerves.  
If a nerve of the brachial plexus is torn beyond any chance of spontaneous recovery, **surgical intervention** is necessary. 
This surgery mainly involves directly suturing the ends of the torn nerve together, but for the surgery to be succesful, the nerve viability must first be assessed by the surgical team. 
To determine the likelihood of recovery and identify the precise locations for suturing, a biopsy and an immediate analysis is thus performed just prior to surgery. 
This is called an **extemporaneous** analysis. 
Although the type of analysis may vary, a common histological technique involves examining sections of the nerve with specific stains to differentiate between **healthy nerve tissue** and **fibrosis**, which is the fibrous connective tissue that results from the scarring process of the torn nerve.  
A common stain used to differentiate between healthy nerve and fibrosis is **HES**  (*hematoxylin, eosin and saffron*). Very briefly, the eosin stains the healthy nerve tissue in pink while the saffron colors the fibrosis in orange. Hematoxilin colors the nucleus in dark blue but it is not relevant for this particular analysis. 
Given the fibrosis rate and the general aspect of the nerve, the surgical team can choose the best locations to stitch the nerves back together and estimate the chances of recovery. 
  
   
<p align="center"><img src="https://github.com/user-attachments/assets/a28f7c5f-a4b1-49d8-9e18-de8b53b2f910" alt="HES" width="600"/><br><i>a) view of a nerve cross-section stained using HES<br>b) the pink/purple part corresponds the the healthy nerve tissue<br>c) the orange part corresponds to fibrosis</i> 
  
The rate of fibrosis is estimated by the histologist, based on his or her best judgment. This is known as a semi-quantitative estimation.  
The **aim of the project** is to provide surgeons with a tool that delivers **fast, repeatable and reliable** rate of fibrosis using the scanned image of the nerve sections. 
In addition, quantitative data are relevant for research into indicators of nerve viability. 


### 2. Dataset  
The dataset consist in a few tens of **whole slide images** (WSI). 
Whole slide imaging (WSI) is a technology that involves scanning entire microscope slides to create high-resolution digital images. 
WSI files use a pyramidal image structure, which means they contain multiple levels of resolution. 
This allows for an efficient navigation through the image at lower resolution and switching to higher resolutions when zooming.  
<p align="center"><img src="https://github.com/user-attachments/assets/f4692eab-4687-4774-be9b-3c075b5731b0" alt="HES" width="800"/><br><i>example of a whole slide image of nerve samples stained with HES, at the lowest resolution</i>  
   
### 3. Approach  

The scarcity of available data rules out a deep learning approach; therefore, we must rely on a purely algorithmic program. 
the first step is to isolate the nerve sections from the noisy background. 
To achieve this, the program first removes irrelevant colors. In the case of HES stain, blue and white can be safely eliminated.
This is accomplished by converting the RGB image into another color space (specifically HLS) and filtering the pixels based on their values in this color space.  

After removing the background, we obtain a large number of blobs. 
However, only a fraction of these blobs corresponds to nerve sections, while the remainder consists of impurities or scratches on the glass. 
These blobs are then filtered based on their size and pixel values to retain only the nerve sections. 
Finally, we open each sample at a higher resolution and separate the pink from the orange to calculate the fibrosis rate for each sample, to ultimately extract an mean value for the whole slide based on each sample's area.  

Since the colors of the stain can vary from one test to another, there cannot be a "one size fits all" color threshold to distinguish between fibrosis and healthy nerve tissue. 
Therefore, a feature has been added to allow the user to fine-tune this threshold while directly observing its effect on the sample image and the fibrosis rate.
