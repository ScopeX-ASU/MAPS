<img src="images/logo.png" title="PreFab" alt="PreFab">

# Important: Time to Use the Shiny New Repository
Hey there!
We've got a brand new repository to replace the old one you've been using—and it's way better! Newer models, design correction, cloud processing, and regular updates. 🎉

Go to [github.com/PreFab-Photonics/PreFab](https://github.com/PreFab-Photonics/PreFab) to get started.

# PreFab
`PreFab` (Prediction of Fabrication) is used for modelling fabrication process induced variations in integrated photonic devices using deep convolutional neural networks.

Trained models predict variations such as corner rounding (both over and under etching), washing away of small lines and islands, and filling of narrow holes and channels in planar photonic structures. Once predicted, the designer resimulates their design to rapidly prototype the expected performance and make any necessary corrections prior to (costly) fabrication.

This repository includes the tools used in the paper `Deep Learning Based Prediction of Fabrication-Process-Induced Structural Variations in Nanophotonic Devices` (which can be viewed [here](https://pubs.acs.org/doi/10.1021/acsphotonics.1c01973)).

![](images/promo.png)
*Predicted fabrication variation of a simple star structure on a 220 nm silicon-on-insulator electron-beam lithography process. Prediction time of 8.2 seconds.*

## Getting Started
Please see the notebooks in `/examples` to get started with making predictions in `PreFab`.

## Models
The models currently included in `/models` are of the NanoSOI process from [Applied Nanotools Inc.](https://www.appliednt.com/nanosoi-fabrication-service/) These are alpha-stage models that are currently in development.

## Authors
`PreFab` was written by Dusan Gostimirovic with Odile Liboiron-Ladouceur (McGill University), Dan-Xia Xu (National Research Council of Canada), and Yuri Grinberg (National Research Council of Canada).

## Usage and Citation
For usage of `PreFab` in your work, please cite our [journal article](https://pubs.acs.org/doi/10.1021/acsphotonics.1c01973).

## License
This project is licensed under the terms of the MIT license. © 2022 National Research Council Canada and McGill University.
