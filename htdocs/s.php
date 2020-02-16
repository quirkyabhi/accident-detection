<!DOCTYPE html>
<html>
<body>
 <?php
    
    $dirname = "img/";
    $images = glob($dirname."*.png") ;

    foreach($images as $image) {
        echo '<img width="400" src="'.$image.'" />';
    }

    ?>
    </body>
</html>