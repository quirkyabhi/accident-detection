<!DOCTYPE html>
<html lang="en">
<head>
	<meta charset="UTF-8">
	<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
	<title>Accident Detection</title>
	<link rel="shortcut icon" href="./imgs/favicon.png">
    <link rel="stylesheet" href="/css/index.css">
</head>
<body>
	<div class="container">
		<!-- <div class="splash-caution">
			Accident
		</div> -->
		<div class="row page-heading">
			<div class="col s12">
				<img src="./imgs/logo.png" />
			</div>
		</div>

		<div id="alert-container">
			<div class="accident-alert">
			    <div class="col s12">
					<h2 >
					    Accident Alert <span>@ 1:20 pm</span>
					</h2>
					<div class="alert-details">
						<h5>
							Location: Sector - 37, Gurgaon
						</h5>
						<p>
							You are <b>10 minutes away</b> from the site of the accident.
						</p>
						<p>
							You are <b>5 kilometres away</b> from the site of the accident.
						</p>
					</div>
			    </div>
			</div>


    
    <?php
    
    $dirname = "img/";
    $images = glob($dirname."*.png") ;

    foreach($images as $image) {
        echo '<img width="800" src="'.$image.'" /><br />';
    }

    ?>

</body>
</html>