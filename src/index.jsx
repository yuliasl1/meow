import React from 'react';
import "@babel/polyfill";
import * as tf from '@tensorflow/tfjs';
import * as mobilenetModule from '@tensorflow-models/mobilenet';
import * as knnClassifier from '@tensorflow-models/knn-classifier';

import { render } from 'react-dom';
import { Provider } from 'react-redux';

import { store } from './_helpers';
import { App } from './App';

// setup fake backend
import { configureFakeBackend } from './_helpers';

// Number of classes to classify
const NUM_CLASSES = 3;
// Webcam Image size. Must be 227.
const IMAGE_SIZE = 237;
// K value for KNN
const TOPK = 10;

const gifs = ['imgs/gangam.gif', 'imgs/eyelash.gif', 'imgs/hi.gif'];

class Train {
    constructor() {
        // Initiate variables
        this.infoTexts = [];
        this.training = -1; // -1 when no class is being trained
        this.videoPlaying = false;
        this.buttonText = ['Dance' , 'Blink', 'Greet']
        this.gifs = ['imgs/gangam.gif', 'imgs/eyelash.gif', 'imgs/hi.gif'];
        // Initiate deeplearn.js math and knn classifier objects
        this.bindPage();

        // Create video element that will contain the webcam image
        this.video = document.createElement('video');
        this.video.setAttribute('autoplay', '');
        this.video.setAttribute('playsinline', '');

        // Add video element to DOM
        document.body.appendChild(this.video);

        this.output = document.getElementById('output');
        // document.body.appendChild(this.output);

        // Create training buttons and info texts
        for (let i = 0; i < NUM_CLASSES; i++) {
            const div = document.createElement('div');
            document.body.appendChild(div);
            div.style.marginBottom = '10px';
            div.style.margin = '7px';
            div.style.padding ='7px';
            div.style.border ='0px';
            // Create training button
            const button = document.createElement('button')
            var element = document.querySelector(".TrainCat");
            button.innerText = "Train me{ow} to " + this.buttonText[i];
            element.appendChild(button);
            //element.insertAdjacentHTML('afterEnd', button)

            // Listen for mouse events when clicking the button
            button.addEventListener('mousedown', () => this.training = i);
            button.addEventListener('mouseup', () => this.training = -1);

            // Create info text
            const infoText = document.createElement('span')
            infoText.innerText = " No examples added";
            var x = document.createElement("BR");
            element.appendChild(x);
            element.appendChild(infoText);
            this.infoTexts.push(infoText);
        }


        // Setup webcam
        navigator.mediaDevices.getUserMedia({ video: true, audio: false })
            .then((stream) => {
                this.video.srcObject = stream;
                this.video.width = IMAGE_SIZE;
                this.video.height = IMAGE_SIZE;

                this.video.addEventListener('playing', () => this.videoPlaying = true);
                this.video.addEventListener('paused', () => this.videoPlaying = false);
            })
    }

    async bindPage() {
        this.knn = knnClassifier.create();
        this.mobilenet = await mobilenetModule.load();

        this.start();
    }

    start() {
        if (this.timer) {
            this.stop();
        }
        this.video.play();
        this.timer = requestAnimationFrame(this.animate.bind(this));
    }

    stop() {
        this.video.pause();
        cancelAnimationFrame(this.timer);
    }

    async animate() {
        if (this.videoPlaying) {
            // Get image data from video element
            const image = tf.browser.fromPixels(this.video);

            let logits;
            // 'conv_preds' is the logits activation of MobileNet.
            const infer = () => this.mobilenet.infer(image, 'conv_preds');

            // Train class if one of the buttons is held down
            if (this.training != -1) {
                logits = infer();

                // Add current image to classifier
                this.knn.addExample(logits, this.training)
            }

            const numClasses = this.knn.getNumClasses();
            if (numClasses > 0) {

                // If classes have been added run predict
                logits = infer();
                const res = await this.knn.predictClass(logits, TOPK);

                for (let i = 0; i < NUM_CLASSES; i++) {

                    // The number of examples for each class
                    const exampleCount = this.knn.getClassExampleCount();

                    // Make the predicted class bold
                    if (res.classIndex === i) {
                        this.infoTexts[i].style.fontWeight = 'bold';
                    } else {
                        this.infoTexts[i].style.fontWeight = 'normal';
                    }
                    //change the output
                    if (res.classIndex === 0 && res.confidences[0]*100 > 90) {
                        document.getElementById('img').src = 'imgs/gangam.gif';
                    }
                    else if (res.classIndex === 1 && res.confidences[1]*100 > 90) {
                        document.getElementById('img').src = 'imgs/eyelash.gif';
                    }
                    else if (res.classIndex === 2 && res.confidences[2]*100 > 90) {
                        document.getElementById('img').src = 'imgs/hi.gif';
                    }
                    else {
                        document.getElementById('img').src = 'imgs/start.gif';
                    }

                    // Update info text
                    if (exampleCount[i] > 0) {
                        this.infoTexts[i].innerText = ` ${exampleCount[i]} examples - ${res.confidences[i] * 100}%`
                    }
                }
            }

            // Dispose image when done
            image.dispose();
            if (logits != null) {
                logits.dispose();
            }
        }
        this.timer = requestAnimationFrame(this.animate.bind(this));
    }
}
configureFakeBackend();

 setTimeout(function(){  new Train();}, 400);
render(
    <Provider store={store}>
        <App />
    </Provider>,
    document.getElementById('app')
);
