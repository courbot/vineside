import scipy.ndimage as ndi
import numpy as np
import matplotlib.pyplot as plt
# import cv2
import skimage.measure as meas
import scipy.spatial as spat
# import os

from skimage.morphology import medial_axis
from skimage.segmentation import slic
from skimage import  color


# ============================================================
# MAIN SEGMENTATION PIPELINE
# ============================================================

def vessel_segmentation_after_sam(img, zoom, area_max, too_close_distance, area_min):
    """
    Post-process segmentation obtained from SAM.

    Steps:
    1. Remove objects touching image borders
    2. Remove clusters of vessels that are too close
    3. Remove vessels that are too small
    4. Remove vessels that are too large

    Parameters
    ----------
    img : ndarray
        Labeled segmentation image produced from SAM masks
    zoom : float
        Image zoom factor (scales spatial parameters)
    area_max : float
        Maximum allowed vessel area
    too_close_distance : float
        Distance threshold for vessel clustering
    area_min : float
        Minimum allowed vessel area

    Returns
    -------
    ndarray
        Cleaned labeled segmentation image
    """

    img_no_border = remove_border(img)

    img_remove_close = remove_too_close(
        img_no_border,
        distance=int(np.ceil(too_close_distance * zoom))
    )

    img_not_too_small = remove_too_small(
        img_remove_close,
        area_min * zoom**2
    )

    img_not_too_big = remove_too_big(
        img_not_too_small,
        2 * area_max * zoom**2
    )

    return img_not_too_big.astype(float)



def vessel_segmentation(image, mask_generator, zoom, area_max,
                        too_close_distance, area_min):
    """
    Complete vessel segmentation pipeline using SAM masks.

    Workflow
    --------
    1. Generate masks using SAM
    2. Convert masks into a labeled segmentation image
    3. Apply morphological filtering

    Parameters
    ----------
    image : ndarray
        Input RGB image
    mask_generator : SAM mask generator
        Object with `.generate()` method
    zoom : float
        Image zoom factor
    area_max : float
        Maximum vessel area
    too_close_distance : float
        Distance threshold for clustered vessels
    area_min : float
        Minimum vessel area

    Returns
    -------
    ndarray
        Cleaned labeled segmentation
    """

    # Generate SAM masks
    masks = mask_generator.generate(image)

    # Convert mask list to label image
    img = make_anns_img(masks)

    # Post-processing
    img_clean = vessel_segmentation_after_sam(
        img,
        zoom,
        area_max,
        too_close_distance,
        area_min
    )

    return img_clean.astype(float)



# ============================================================
# FILTERING FUNCTIONS
# ============================================================

def remove_border(img):
    """
    Remove objects touching the image borders.

    Parameters
    ----------
    img : ndarray
        Labeled segmentation image

    Returns
    -------
    ndarray
        Image with border-touching objects removed
    """

    labels = np.unique(img)
    img_out = np.zeros_like(img)

    for lab in labels:

        mask = img == lab

        col_sum = mask.sum(axis=0)
        row_sum = mask.sum(axis=1)

        border_pixels = (
            col_sum[0] +
            col_sum[-1] +
            row_sum[0] +
            row_sum[-1]
        )

        if border_pixels == 0:
            img_out[mask] = lab

    return img_out


def remove_too_close(img_filter, distance, max_connect=5):
    """
    Remove clusters of vessels that are too close together.

    Morphological closing merges nearby vessels. Clusters with
    more than `max_connect` connected vessels are removed.

    Parameters
    ----------
    img_filter : ndarray
        Labeled segmentation
    distance : int
        Closing radius
    max_connect : int
        Maximum number of vessels allowed in a cluster

    Returns
    -------
    ndarray
        Filtered segmentation
    """

    mask_bin = img_filter > 0

    # morphological closing merges nearby objects
    closed = ndi.binary_closing(mask_bin, iterations=distance)

    # second smoothing step
    closed = ndi.binary_closing(mask_bin, iterations=3)

    closed_labels, n_labels = ndi.label(closed)

    out = np.zeros_like(closed_labels)

    for label_id in range(n_labels):

        mask_cluster = closed_labels == label_id
        labels_inside = np.unique(mask_cluster * img_filter)

        n_connected = labels_inside.size - 1

        areas = np.zeros_like(labels_inside)

        for i in range(n_connected + 1):
            areas[i] = (img_filter == labels_inside[i]).sum()

        if n_connected <= max_connect:
            for lab in labels_inside[1:]:
                out[img_filter == lab] = lab

    return out.astype(float)


def remove_too_far(img):
    """
    Remove objects spatially isolated from the main cluster.

    Objects whose average distance to all others exceeds
    mean + std of distances are removed.
    """

    prop = meas.regionprops(img.astype(int))
    n_obj = len(prop)

    positions = np.array([p.centroid for p in prop])

    D = spat.distance_matrix(positions, positions)

    Dmean = D.mean(axis=0)

    remove = Dmean > Dmean.mean() + Dmean.std()

    img_out = np.zeros_like(img)

    for i in range(n_obj):

        pos = positions[i].astype(int)
        lab = img[pos[0], pos[1]]

        if not remove[i]:
            mask = img == lab
            img_out[mask] = i

    return remove_border(img_out)


def remove_too_small(img, min_size):
    """
    Remove objects smaller than a given area.
    """

    img_out = np.copy(img)

    for lab in np.unique(img):

        if (img == lab).sum() < min_size:
            img_out[img == lab] = 0

    return img_out


def remove_too_big(img, max_size):
    """
    Remove objects larger than a given area.
    """

    img_out = np.copy(img)

    for lab in np.unique(img):

        if (img == lab).sum() > max_size:
            img_out[img == lab] = 0

    return img_out

# ============================================================
# SAM UTILITIES
# ============================================================

def make_anns_img(anns):
    """
    Convert SAM annotations into a labeled image.

    Masks are sorted by decreasing area so large masks
    overwrite smaller ones.

    Parameters
    ----------
    anns : list
        SAM annotations

    Returns
    -------
    ndarray
        Labeled segmentation image
    """

    anns_sorted = sorted(anns, key=lambda x: x['area'], reverse=True)

    H, W = anns_sorted[0]['segmentation'].shape
    img = np.zeros((H, W))

    label = 1

    for ann in anns_sorted:

        mask = ann['segmentation']
        img[mask] = label
        label += 1

    return img


def filter_anns(anns, low, high):
    """
    Filter SAM masks by area.
    """

    anns_sorted = sorted(anns, key=lambda x: x['area'], reverse=True)

    H, W = anns_sorted[0]['segmentation'].shape
    img = np.zeros((H, W))

    label = 1

    for ann in anns_sorted:

        if low < ann['area'] < high:

            mask = ann['segmentation']
            img[mask] = label
            label += 1

    return img
# ============================================================
# VISUALIZATION
# ============================================================

def display(image, image_label):
    """
    Overlay segmentation labels on the original image.
    """

    plt.imshow(image.mean(axis=2), alpha=0.25, cmap='gray')

    overlay = image_label.copy()
    overlay[overlay == 0] = np.nan

    plt.imshow(overlay, alpha=0.75, cmap='Spectral')

    plt.axis('off')


# ============================================================
# POLAR MEASUREMENTS
# ============================================================

def get_area_count(final_segmentation, mask_new_xylem):
    """
    Compute polar statistics of vessel properties.

    Returns:
        polar area
        eccentricity
        vessel density
    """

    def get_center(img):

        prop = meas.regionprops(img.astype(int))
        positions = np.array([p.centroid for p in prop])

        return positions.mean(axis=0)

    def get_max_radius(img, mask):

        only_new_xylem = img * mask

        if only_new_xylem.sum() > 0:

            prop = meas.regionprops(only_new_xylem.astype(int))
            pos = np.array([p.centroid for p in prop])

            radius = np.sqrt(((pos - center) ** 2).sum(axis=1))

            return radius.max()

        return np.inf


    N_theta = 33

    center = get_center(final_segmentation)

    max_radius = get_max_radius(final_segmentation, mask_new_xylem)

    img_no_xylem = final_segmentation * (mask_new_xylem == 0)

    prop = meas.regionprops(img_no_xylem.astype(int))

    areas = np.array([p.area for p in prop])
    positions = np.array([p.centroid for p in prop])
    ecc = np.array([p.eccentricity for p in prop])

    center = positions.mean(axis=0)

    radius = np.sqrt(((positions - center) ** 2).sum(axis=1))
    angle = np.arctan2(positions[:, 0] - center[0],
                       positions[:, 1] - center[1])

    mask_in = radius < max_radius

    theta = np.linspace(-np.pi, np.pi, N_theta)

    pol_areas = np.zeros(N_theta - 1)
    pol_ecc = np.zeros(N_theta - 1)

    for i in range(N_theta - 1):

        mask = (
            (angle[mask_in] > theta[i]) &
            (angle[mask_in] < theta[i+1])
        )

        pol_areas[i] = areas[mask_in][mask].mean()
        pol_ecc[i] = ecc[mask_in][mask].mean()

    # physical conversion
    pix_per_um = 0.22
    um2_per_pix = 1 / pix_per_um**2

    pol_areas *= um2_per_pix

    count = np.zeros(N_theta - 1)

    for i in range(N_theta - 1):

        count[i] = (
            (angle[mask_in] > theta[i]) &
            (angle[mask_in] < theta[i+1])
        ).sum()

    count /= count.sum()

    return pol_areas, pol_ecc, count


def display_polar_measurements(areas, count, ecc):
    """
    Plot polar distributions of vessel properties.
    """

    plt.figure(figsize=(12, 4))

    N_theta = 33
    theta = np.linspace(-np.pi, np.pi, N_theta)
    theta += (theta[1] - theta[0]) / 2

    ax1 = plt.subplot(131, projection='polar') ; ax1.set_rlabel_position(90) ; ax1.set_rmin(1000);ax1.set_rmax(9000)
    ax2 = plt.subplot(132, projection='polar') ; ax2.set_rlabel_position(90)
    ax3 = plt.subplot(133, projection='polar') ; ax3.set_rlabel_position(90)

    ax1.set_title("Xylem vessel area (µm²)")
    ax2.set_title("Xylem vessel density")
    ax3.set_title("Xylem vessel eccentricity")

    ax1.plot(theta, np.append(areas, areas[0]),color='g')
    ax2.plot(theta, np.append(count, count[0]),color='g')
    ax3.plot(theta, np.append(ecc, ecc[0]),color='g')

    plt.tight_layout()
    
   

# ============================================================
# Measure phloem width
# ============================================================


def get_phloem_median_width(image,filter_size=10):
    """
    Estimate the median width of the phloem region.

    Pipeline
    --------
    1. Convert image to grayscale
    2. Threshold to isolate tissue regions
    3. Keep the largest relevant connected component
    4. Apply morphological closing and hole filling
    5. Extract phloem region (complementary structure)
    6. Compute medial axis distance transform
    7. Estimate width as median diameter

    Parameters
    ----------
    image : ndarray (H, W, 3)
        Input RGB image
    filter_size : float
        Gaussian smoothing parameter for mask regularization

    Returns
    -------
    median_width : float
        Estimated phloem width (in pixels)
    mask_phloem : ndarray (bool)
        Binary mask of the phloem region
    """
    
    def get_second_largest_component(mask):
        """
        Extract the second largest connected component from a binary mask.

        Note:
        Largest component is often background; second largest corresponds
        to the main biological structure of interest.
        """
        lab,nlab = ndi.label(mask==0)
        areas = np.array([(lab==k).sum() for k in range(nlab)])
        i_sort = np.argsort(areas)
        ind_bigger = i_sort[-2]
        mask_bigger = lab==ind_bigger
        return mask_bigger
    
    # Zoom the image to work on an under-resolved version, to accelerate computations.
    zoom_factor = 4
    image_dezoom = ndi.zoom(image,(1/zoom_factor,1/zoom_factor,1))
    
    # Over-segmentation based on SLIC super-pixels.
    segments_slic = slic(image_dezoom, n_segments=600, compactness=50, sigma=1, start_label=0)
    N_segments = segments_slic.max()
    
    im_slic = np.zeros_like(image_dezoom)
    for k in range(N_segments):
        mask = segments_slic==k
        for c in (0,1,2):
            im_slic[mask,c] = image_dezoom[mask,c].mean()
    
    # Thresholding: darker areas are the phloem.
    im_gris = im_slic.mean(axis=2)
    mask = im_gris > im_gris.mean()-im_gris.std()
    msk_phloem = get_second_largest_component(mask)
    
    # Width computation
    skel, distance = medial_axis(msk_phloem, return_distance=True)
    dist = distance[skel!=0]*1.0
    median_width= np.median(dist)*2 # width = 2*distance from central axis

    return median_width*zoom_factor,ndi.zoom(msk_phloem,(zoom_factor,zoom_factor),order=0)



def get_colors(image, max_L=90):
    """
    Extract representative colors from an image using SLIC superpixels.

    Pipeline
    --------
    1. Downsample the image (speed + denoising)
    2. Convert to CIELAB color space
    3. Segment image into superpixels (SLIC)
    4. Compute mean color per segment (RGB + LAB)
    5. Filter out low-variance (gray/black/white-like) segments

    Parameters
    ----------
    image : ndarray (H, W, 3)
        Input RGB image
    max_L : float, optional
        Unused parameter (kept for compatibility)

    Returns
    -------
    couleurs_lab : ndarray (N, 3)
        Mean LAB colors of selected segments
    couleurs_rgb : ndarray (N, 3)
        Mean RGB colors of selected segments
    im_slic : ndarray (H', W', 3)
        Image where each segment is replaced by its mean color
    """

    # ========================================================
    # 1. Downsample image (reduce computation + smooth noise)
    # ========================================================
    image_small = ndi.zoom(image, (0.25, 0.25, 1))

    # ========================================================
    # 2. Convert to LAB color space (perceptual space)
    # ========================================================
    image_lab = color.rgb2lab(image_small)

    # ========================================================
    # 3. SLIC superpixel segmentation
    # ========================================================
    segments = slic(
        image_small,
        n_segments=600,
        compactness=20,
        sigma=1,
        start_label=0
    )

    n_segments = segments.max() + 1

    # ========================================================
    # 4. Compute mean color per segment
    # ========================================================
    couleurs_lab = np.zeros((n_segments, 3))
    couleurs_rgb = np.zeros((n_segments, 3))

    # Reconstructed image (each segment replaced by mean color)
    im_slic = np.zeros_like(image_small)

    for k in range(n_segments):

        mask = segments == k

        # Mean color per channel
        mean_rgb = image_small[mask].mean(axis=0)
        mean_lab = image_lab[mask].mean(axis=0)

        couleurs_rgb[k] = mean_rgb
        couleurs_lab[k] = mean_lab

        # Fill reconstructed image
        im_slic[mask] = mean_rgb

    # ========================================================
    # 5. Remove low-variance (gray-like) segments
    # ========================================================
    # Heuristic: low std across RGB channels → near grayscale
    color_std = couleurs_rgb.std(axis=1)

    mask_valid = color_std > 0.05

    return couleurs_lab[mask_valid], couleurs_rgb[mask_valid], im_slic

def get_new_xylem_area_number(mask_new_xylem, final_segmentation):
    """
    Compute statistics of vessels fully contained in the new xylem region.

    Method
    ------
    1. Intersect vessel segmentation with the new xylem mask
    2. Remove vessels that are only partially داخل the new xylem
    3. Label remaining vessels
    4. Compute their areas
    5. Convert areas to physical units
    6. Return median area and number of vessels

    Parameters
    ----------
    mask_new_xylem : ndarray (bool or int)
        Binary mask defining the new xylem region
    final_segmentation : ndarray (int)
        Labeled vessel segmentation (each vessel has a unique label)

    Returns
    -------
    median_area : float
        Median vessel area (µm²)
    n_vessels : int
        Number of vessels fully داخل the new xylem
    """

    # ========================================================
    # 1. Intersection: keep vessels inside new xylem
    # ========================================================
    intersection = mask_new_xylem * final_segmentation

    # Copy to progressively remove invalid vessels
    vessels_in_xylem = np.copy(intersection)

    labels = np.unique(intersection)


    # ========================================================
    # 2. Remove partially intersecting vessels
    # ========================================================
    # Keep only vessels entirely contained in the new xylem mask
    for lab in labels:

        if lab == 0:
            continue  # skip background

        total_area = (final_segmentation == lab).sum()
        intersect_area = (intersection == lab).sum()

        # If not fully داخل the mask → remove
        if total_area != intersect_area:
            vessels_in_xylem[intersection == lab] = 0


    # ========================================================
    # 3. Relabel connected components
    # ========================================================
    labeled_vessels, _ = ndi.label(vessels_in_xylem)


    # ========================================================
    # 4. Compute vessel areas
    # ========================================================
    props = meas.regionprops(labeled_vessels.astype(int))

    areas = np.array([p.area for p in props])


    # ========================================================
    # 5. Unit conversion (pixels → µm²)
    # ========================================================
    pix_per_um = 0.22
    um2_per_pix = 1.0 / pix_per_um**2

    areas = areas * um2_per_pix


    # ========================================================
    # 6. Output statistics
    # ========================================================
    median_area = np.median(areas) if areas.size > 0 else np.nan
    n_vessels = areas.size

    return median_area, n_vessels

def get_phloem_width(mask_new_xylem):
    """
    Estimate phloem width using medial axis transform.

    Method
    ------
    The medial axis provides, at each skeleton point, the distance
    to the nearest boundary (i.e., local radius). The width is then
    approximated as:

        width ≈ 2 × median(radius)

    The median is used for robustness against local irregularities.

    Parameters
    ----------
    mask_new_xylem : ndarray (bool or int)
        Binary mask of the xylem region

    Returns
    -------
    float
        Estimated phloem width (in micrometers)
    """

    # ========================================================
    # 1. Medial axis and distance transform
    # ========================================================
    skel, dist = medial_axis(mask_new_xylem, return_distance=True)

    # Extract radius values only on the skeleton
    radii = dist[skel]

    # ========================================================
    # 2. Unit conversion (pixels → micrometers)
    # ========================================================
    pix_per_um = 0.22
    um_per_pix = 1.0 / pix_per_um

    # ========================================================
    # 3. Width estimation
    # ========================================================
    # width ≈ 2 × median radius
    width = 2.0 * np.median(radii) * um_per_pix

    return width
