$(document).ready(function() {
    $('.image-section').hide();
    $('.center').hide();
    $('#result').hide();

    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function(e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    
    $("#imageUpload").change(function() {
        var categoryInput = $('#cc').val().trim();
        var driveLinkInput = $('#dl').val().trim(); 
        
    var driveLinkPattern = /^https:\/\/drive\.google\.com\/drive\/.*$/;

       if (categoryInput === '' || driveLinkInput === '') {
        alert('Please fill in both Classification Category Name and Drive Link.');
        window.location.href = '/'; 
        return; 
            }
          else if (!driveLinkPattern.test(driveLinkInput)) {
        alert('Please enter a valid Google Drive link in the format "https://drive.google.com/drive/folders/...".');
                window.location.href = '/'; 

        return; 

    } else {
        $('#imageUpload').prop('disabled', false); 
    }
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    $('#btn-predict').click(function() {
        var form_data = new FormData($('#upload-file')[0]);

        $(this).hide();
        $('.center').show();

        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function(data) {
                var lines = data.split('\n');

                $('.center').hide();
                $('#result').fadeIn(600);

                for (var i = 0; i < lines.length; i++) {
                    var line = lines[i];
                 if(i==4){
                    $('#result').append(line);
                 }
                else{
                 $('#result').append('<p><h2>' + line + '</h2></p>');
                 }
                }

                console.log('Success!');
            },
        });
    });

});