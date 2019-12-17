import cv2
import gphoto2 as gp
import sys
import io
from PIL import Image
import numpy as np

camera = gp.check_result( gp.gp_camera_new() )
gp.check_result( gp.gp_camera_init( camera ) )

for i in range( 0, 15 ):
    file_path = gp.check_result( gp.gp_camera_capture( camera, gp.GP_CAPTURE_IMAGE ) )
    camera_file = gp.check_result( gp.gp_camera_file_get( camera, file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL ) )
    file_data = gp.check_result(gp.gp_file_get_data_and_size(camera_file))

    image = Image.open(io.BytesIO(file_data))
    gp.check_result(gp.gp_camera_exit(camera))
    img = cv2.cvtColor( np.array( image ), cv2.COLOR_RGB2BGR )

    # DOES NOT WORK YET
    #file_bytes = np.asarray(bytearray(io.BytesIO( file_data ).read()), dtype=np.uint8)
    #img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    cv2.imwrite( "test" + str( i ) + ".jpg", img )