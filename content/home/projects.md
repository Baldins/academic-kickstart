+++
# A Projects section created with the Portfolio widget.
widget = "portfolio"  # See https://sourcethemes.com/academic/docs/page-builder/
headless = true  # This file represents a page section.
active = true  # Activate this widget? true/false
weight = 65  # Order that this section will appear.

title = "Learning Pose Estimation for UAV Autonomous Navigation and Landing Using Visual-Inertial Sensor Data"
subtitle = ""
summary = "In this work, we propose a new learning approach for autonomous navigation and landing of an Unmanned-Aerial-Vehicle (UAV). We develop a multimodal fusion of deep neural architectures for visual-inertial odometry. We train the model in an end-to-end fashion to estimate the current vehicle pose from streams of visual and inertial measurements.
We first evaluate the accuracy of our estimation by comparing the prediction of the model to traditional algorithms on the publicly available EuRoC MAV dataset. The results illustrate a $25 \%$ improvement in estimation accuracy over the baseline. Finally, we integrate the architecture in the closed-loop flight control system of Airsim - a plugin simulator for Unreal Engine - and we provide simulation results for autonomous navigation and landing."
[content]
  # Page type to display. E.g. project.
  page_type = "project"
  
  # Filter toolbar (optional).
  # Add or remove as many filters (`[[content.filter_button]]` instances) as you like.
  # To show all items, set `tag` to "*".
  # To filter by a specific tag, set `tag` to an existing tag name.
  # To remove toolbar, delete/comment all instances of `[[content.filter_button]]` below.
  
  # Default filter index (e.g. 0 corresponds to the first `[[filter_button]]` instance below).
  filter_default = 0
  
  # [[content.filter_button]]
  #   name = "All"
  #   tag = "*"
  
  # [[content.filter_button]]
  #   name = "Deep Learning"
  #   tag = "Deep Learning"
  
  # [[content.filter_button]]
  #   name = "Other"
  #   tag = "Demo"

[design]
  # Choose how many columns the section has. Valid values: 1 or 2.
  columns = "1"

  # Toggle between the various page layout types.
  #   1 = List
  #   2 = Compact
  #   3 = Card
  #   5 = Showcase
  view = 3

  # For Showcase view, flip alternate rows?
  flip_alt_rows = false

[design.background]
  # Apply a background color, gradient, or image.
  #   Uncomment (by removing `#`) an option to apply it.
  #   Choose a light or dark text color by setting `text_color_light`.
  #   Any HTML color name or Hex value is valid.
  
  # Background color.
  <!-- # color = "navy" -->
  
  # Background gradient.
  <!-- # gradient_start = "DeepSkyBlue" -->
  # gradient_end = "SkyBlue"
  
  # Background image.
  # image = "background.jpg"  # Name of image in `static/img/`.
  # image_darken = 0.6  # Darken the image? Range 0-1 where 0 is transparent and 1 is opaque.

  # Text color (true=light or false=dark).
  <!-- # text_color_light = true   -->
  
[advanced]
 # Custom CSS. 
 css_style = ""
 
 # CSS class.
 css_class = ""
+++

