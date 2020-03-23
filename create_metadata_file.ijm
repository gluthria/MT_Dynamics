dir = getDirectory("Choose a Directory");
file = File.open(dir + "image_resolution_framerate.csv");
for (i=0;i<nImages;i++) {
        selectImage(i+1);
        frame_interval = Stack.getFrameInterval();
        getPixelSize (unit, pixelWidth, pixelHeight);
        resolution = 1/pixelWidth;
        title = getTitle;
        title_small = split(title,".");
        //print(title_small[0]+".csv," + resolution + "," + frame_interval);
        print(file, title_small[0]+".csv," + resolution + "," + frame_interval + "\n");
        //saveAs("tiff", dir+title);
}
run("Close All");