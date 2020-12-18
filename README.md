# 2D Feature Tracking

## Steps

### MP.1 Data Buffer Optimization

Implement a vector for dataBuffer objects whose size does not exceed 3.

```
if (dataBuffer.size() > dataBufferSize)
{
    dataBuffer.erase(dataBuffer.begin());
}
```

### MP.2 Keypoint Detection

Implement detectors: Harris, FAST, BRISK, ORB, AKAZE, and SIFT. Make them selectable by setting a string accordingly.

```
cv::Ptr<cv::FeatureDetector> detector;

if (detectorType.compare("FAST") == 0)
{
    int threshold = 30;
    bool nonmaxSuppression = true;
    cv::FAST(img, keypoints, threshold, nonmaxSuppression, cv::FastFeatureDetector::TYPE_9_16);
}
else if (detectorType.compare("BRISK") == 0)
{
    detector = cv::BRISK::create();
}
else if (detectorType.compare("ORB") == 0)
{
    detector = cv::ORB::create();
}
else if (detectorType.compare("AKAZE") == 0)
{
    detector = cv::AKAZE::create();
}
else if (detectorType.compare("SIFT") == 0)
{
    detector = cv::SIFT::create();
}

detector->detect(img, keypoints);
````

### MP.3 Keypoint Removal

Remove all keypoints outside a pre-defined rectangle and only use the keypoints within the rectangle for further processing.

```
bool bFocusOnVehicle = true;
cv::Rect vehicleRect(535, 180, 180, 150);
if (bFocusOnVehicle)
{
    for (size_t i = 0; i < keypoints.size(); i++)
    {
        if (!vehicleRect.contains(keypoints[i].pt))
        {
            keypoints.erase(keypoints.begin() + i);
            i--;
        }
    }
}
```

### MP.4 Keypoint Descriptors

Implement descriptors BRIEF, ORB, FREAK, AKAZE, AND SIFT. Make them selectable by setting a string accordingly.

```
cv::Ptr<cv::DescriptorExtractor> extractor;
if (descriptorType.compare("BRISK") == 0)
{

    int threshold = 30;        // FAST/AGAST detection threshold score.
    int octaves = 3;           // detection octaves (use 0 to do single scale)
    float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

    extractor = cv::BRISK::create(threshold, octaves, patternScale);
}
else if (descriptorType.compare("ORB") == 0)
{
    extractor = cv::ORB::create();
}
else if (descriptorType.compare("BRIEF") == 0)
{
    extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
}
else if (descriptorType.compare("SIFT") == 0)
{
    extractor = cv::SIFT::create();
}
else if (descriptorType.compare("FREAK") == 0)
{
    extractor = cv::xfeatures2d::FREAK::create();
}
else if (descriptorType.compare("AKAZE") == 0)
{
    extractor = cv::AKAZE::create();
}

extractor->compute(img, keypoints, descriptors);
```

### MP.5 Descriptor Matching

Implement FLANN based matching.

```
bool crossCheck = false;
cv::Ptr<cv::DescriptorMatcher> matcher;

if (matcherType.compare("MAT_BF") == 0)
{
    int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
    matcher = cv::BFMatcher::create(normType, crossCheck);
}
else if (matcherType.compare("MAT_FLANN") == 0)
{
    if (descSource.type() != CV_32F)
    {
        descSource.convertTo(descSource, CV_32F);
        descRef.convertTo(descRef, CV_32F);
    }
    matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
}
```

### MP.6 Descriptor Distance Ratio

Implement K-nearest-neighbor matching with distance ratio test, which looks at the ratio of best vs. second-best match to decide whether to keep and associated pare of keypoints.

```
if (selectorType.compare("SEL_NN") == 0)
{ // nearest neighbor (best match)

    matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
}
else if (selectorType.compare("SEL_KNN") == 0)
{ // k nearest neighbors (k=2)

    int k = 2;
    double minDistanceRatio = 0.8;
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(descSource, descRef, knn_matches, k);

    for (auto it = knn_matches.begin(); it != knn_matches.end(); it++)
    {
    if ((*it)[0].distance < minDistanceRatio * (*it)[1].distance)
        {
            matches.push_back((*it)[0]);
        }
    }
}
```

### MP.7 Performance Evaluation 1

#### Keypoints Counting

The table shows the average keypoints amount for different detectors. Harris corner detector has the smallest amount of keypoints while BRISK has the most amount of keypoints.

|Detector|Average keypoints amount|
|--------|:----------------------:|
|Harris|**24.8**|
|Shi-Tomasi|117.9|
|FAST|149.1|
|BRISK|**276.2**|
|ORB|116.1|
|AKAZE|167.7|
|SIFT|138.6|

#### Distribution of neighborhood

###### Harris

###### Shi-Tomasi

###### FAST

###### BRISK

###### ORB

###### AKAZE

###### SIFT

### MP.8 Performance Evaluation 2

Count the number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors. In the matching step, the BF approach is used with the descriptor distance ratio set to 0.8.

#### Number of Matched Key Points

|Number of Key Points|Average|Number||||
|---|:---:|:---:|:---:|:---:|:---:|
|Detector / Descriptor|ORB|AKAZE|SIFT|BRIEF|FREAK|
|ORB|84.6|x|84.8|60.6|46.8|
|AKAZE|131.8|139.9|141.1|140.7|132.0|
|SIFT|x|x|88.9|78.0|66.2|
|Harris|17.8|x|18.1|19.2|**16.2**|
|Shi-Tomasi|100.8|x|103.0|106.9|85.1|
|FAST|120.1|x|116.2|122.1|97.9|
|BRISK|167.8|x|182.9|**189.3**|169.6|

### MP.9 Performance Evaluation 3

Log the time it takes for keypoint detection adn descriptor extraction.

###### Note
- Binary descriptors: BRISK, BRIEF, ORB, FREAK, and AKAZE
    using NORM_HAMMING
- HOG descriptors: SIFT
    using NORM_L2
- NORM_HAMMING2 should be used with ORB when WTA_K == 3 or 4
- The following combination will give you error:
  - KAZE/AKAZE descriptors will only work with KAZE/AKAZE detectors
  - SIFT detector and ORB descriptor do not work together
- The BF approach is used with the descriptor distance ratio set to 0.8
- Only those keypoints on the preceding vehicle in image are processed

ORB detector with BRIEF descriptor will give us the fastest running time while BRISK detector with FREAK descriptor will give us the slowest result.

#### Total Time For Keypoint Detection and Descriptor Extraction

|Total Time|Average|Time|(ms)|||
|---|:---:|:---:|:---:|:---:|:---:|
|Detector / Descriptor|ORB|AKAZE|SIFT|BRIEF|FREAK|
|ORB|21.8|x|45.4|**9.0**|42.5|
|AKAZE|86.4|128.5|94.9|76.4|107.4|
|SIFT|x|x|162.5|101.1|132.5|
|Harris|18.5|x|30.7|16.0|46.8|
|Shi-Tomasi|28.0|x|41.2|26.1|58.6|
|FAST|28.5|x|43.8|25.1|58.5|
|BRISK|333.8|x|354.3|321.5|**355.6**|

### Conclusion

By considering all of these variations, My top three detector/descriptor combinations are:
1. FAST + ORB
2. FAST + BRIEF
3. Shi-Tomasi + BRIEF
