(function() {
  var layerGroup = {{ this.data.layer }};
  var payload = {{ this.data.payload | script_json }};

  async function decompress(base64Str) {
    var binary = atob(base64Str);
    var bytes = new Uint8Array(binary.length);
    for (var i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    var stream = new Response(bytes).body.pipeThrough(new DecompressionStream('gzip'));
    return await new Response(stream).json();
  }

  decompress(payload).then(function(geoJsonData) {
    L.geoJson(geoJsonData, {
      style: {
        fill: false,
        fillOpacity: 0,
        color: "black",
        weight: 2
      }
    }).addTo(layerGroup);
  });
})();
