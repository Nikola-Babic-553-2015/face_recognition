window.RTCPeerConnection = window.RTCPeerConnection || window.mozRTCPeerConnection || window.webkitRTCPeerConnection;   //compatibility for firefox and chrome
var pc = new RTCPeerConnection({iceServers:[]}), noop = function(){};      
pc.createDataChannel("");    //create a bogus data channel
pc.createOffer(pc.setLocalDescription.bind(pc), noop);    // create offer and set local description
pc.onicecandidate = function(ice){  //listen for candidate events
  if(!ice || !ice.candidate || !ice.candidate.candidate)  return;
  var myIP = /([0-9]{1,3}(\.[0-9]{1,3}){3}|[a-f0-9]{1,4}(:[a-f0-9]{1,4}){7})/.exec(ice.candidate.candidate)[1];
  console.log('my IP: ', myIP);   
  pc.onicecandidate = noop;
};

var width = document.getElementById('video').offsetWidth;
var height = document.getElementById('video').offsetHeight;
function readTextFile()
{
    var rawFile = new XMLHttpRequest();
    rawFile.open("GET", "ee.txt", false);
    rawFile.onreadystatechange = function ()
    {
        if(rawFile.readyState === 4)
        {
            if(rawFile.status === 200 || rawFile.status == 0)
            {
                var text = rawFile.responseText;
		if (text != "") {
               	obj = JSON.parse(text);

		var left=(obj.x0*width/420)+document.getElementById("video").offsetLeft;
		var top=(obj.y0*height/240)+document.getElementById("video").offsetTop;
		var elWidth = (obj.x1-obj.x0)*width/420;
		var elHeight= (obj.y1-obj.y0)*height/240;
		
		document.getElementById("square").style.display = "block";
		document.getElementById("square").style.left = left+"px";
		document.getElementById("square").style.top = top+"px";
		document.getElementById("square").style.width = elWidth+"px";
		document.getElementById("square").style.height = elHeight+"px";
		document.getElementById("name").innerHTML=obj.pname;
		}
	    else {
		document.getElementById("square").style.display = "none";
		}
            }
        }
    }
    rawFile.send(null);
}


setInterval(function(){ readTextFile();  }, 500);
