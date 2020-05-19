import {
  AmbientLight,
  AnimationMixer,
  AxesHelper,
  Box3,
  Cache,
  CubeTextureLoader,
  DirectionalLight,
  GridHelper,
  HemisphereLight,
  LinearEncoding,
  LoaderUtils,
  LoadingManager,
  PMREMGenerator,
  PerspectiveCamera,
  RGBFormat,
  Scene,
  SkeletonHelper,
  UnsignedByteType,
  Vector3,
  WebGLRenderer,
  sRGBEncoding,
  // nfrechette - BEGIN
  PropertyBinding,
  ZeroCurvatureEnding,
  InterpolateLinear,
  InterpolateSmooth,
  InterpolateDiscrete,
  LinearInterpolant,
  DiscreteInterpolant,
  CubicInterpolant,
  QuaternionLinearInterpolant,
  GLTFCubicSplineInterpolant,
  // nfrechette - END
} from 'three';
import Stats from 'three/examples/jsm/libs/stats.module.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { RGBELoader } from 'three/examples/jsm/loaders/RGBELoader.js';
// import { RoughnessMipmapper } from 'three/examples/jsm/utils/RoughnessMipmapper.js';

import { GUI } from 'dat.gui';

import { environments } from '../assets/environment/index.js';
import { createBackground } from '../lib/three-vignette.js';

// nfrechette - BEGIN
import { TrackArray, SampleTypes, QVV, RoundingPolicy, Encoder } from '@nfrechette/acl';
import { DecompressedTracks, Decoder } from '@nfrechette/acl';
// nfrechette - END

const DEFAULT_CAMERA = '[default]';

const IS_IOS = /iPad|iPhone|iPod/.test(navigator.userAgent) && !window.MSStream;

// glTF texture types. `envMap` is deliberately omitted, as it's used internally
// by the loader but not part of the glTF format.
const MAP_NAMES = [
  'map',
  'aoMap',
  'emissiveMap',
  'glossinessMap',
  'metalnessMap',
  'normalMap',
  'roughnessMap',
  'specularMap',
];

const Preset = {ASSET_GENERATOR: 'assetgenerator'};

Cache.enabled = true;

// nfrechette - BEGIN
// This uses the raw ACL tracks to interpolate and playback
// This is slow and for debugging purposes only
class ACLRawInterpolant {
  constructor(clip, track, result) {
    this.clip = clip
    this.track = track
    this.resultBuffer = result
    this.numOutputValues = track.numBlendWeightTracks
  }

  evaluate(t) {
    if (this.track.propertyName === 'morphTargetInfluences') {
      for (let i = 0; i < this.numOutputValues; ++i) {
        const trackIndex = this.track.aclTrackOffset + i
        this.resultBuffer[i] = this.clip.aclWeights.raw.sampleTrack(trackIndex, t, RoundingPolicy.None)
      }
    }
    else {
      const transform = this.clip.aclTransforms.raw.sampleTrack(this.track.aclTrackIndex, t, RoundingPolicy.None)

      if (this.track.propertyName === 'quaternion') {
        this.resultBuffer[0] = transform.rotation.x
        this.resultBuffer[1] = transform.rotation.y
        this.resultBuffer[2] = transform.rotation.z
        this.resultBuffer[3] = transform.rotation.w
      }
      else if (this.track.propertyName === 'position') {
        this.resultBuffer[0] = transform.translation.x
        this.resultBuffer[1] = transform.translation.y
        this.resultBuffer[2] = transform.translation.z
      }
      else if (this.track.propertyName === 'scale') {
        this.resultBuffer[0] = transform.scale.x
        this.resultBuffer[1] = transform.scale.y
        this.resultBuffer[2] = transform.scale.z
      }
    }

    return this.resultBuffer
  }
}

// This uses the compressed ACL tracks to interpolate and playback
class ACLInterpolant {
  constructor(clip, decoder, track, result) {
    if (track.propertyName === 'morphTargetInfluences') {
      this.compressedTracks = clip.aclWeights.compressed
      this.decompressedTracks = clip.aclWeights.decompressed
      this.numOutputValues = track.numBlendWeightTracks
      this.inputOffset = track.aclTrackOffset
    }
    else {
      this.compressedTracks = clip.aclTransforms.compressed
      this.decompressedTracks = clip.aclTransforms.decompressed

      let numOutputValues = 0
      let inputOffset = 0
      if (track.propertyName === 'quaternion') {
        numOutputValues = 4
        inputOffset = 0
      }
      else if (track.propertyName === 'position') {
        numOutputValues = 3
        inputOffset = 4
      }
      else if (track.propertyName === 'scale') {
        numOutputValues = 3
        inputOffset = 8
      }

      this.numOutputValues = numOutputValues
      this.inputOffset = track.aclTrackIndex * 12 + inputOffset // 12 floats per QVV
    }

    this.decoder = decoder
    this.resultBuffer = result
  }

  evaluate(t) {
    if (this.decompressedTracks.cachedTime != t) {
      this.decoder.decompressTracks(this.compressedTracks, t, RoundingPolicy.None, this.decompressedTracks)
      this.decompressedTracks.cachedTime = t
    }

    const array = this.decompressedTracks.array
    const inputOffset = this.inputOffset
    for (let i = 0; i < this.numOutputValues; ++i) {
      this.resultBuffer[i] = array[inputOffset + i]
    }

    return this.resultBuffer
  }
}

// This uses the compressed ACL tracks to interpolate and playback
// Unlike ACLInterpolant, it will decompress each track individually
// This is slow and for debugging purposes only
class ACLPerTrackInterpolant {
  constructor(clip, decoder, track, result) {
    if (track.propertyName === 'morphTargetInfluences') {
      this.compressedTracks = clip.aclWeights.compressed
      this.decompressedTracks = clip.aclWeights.decompressed
      this.numOutputValues = track.numBlendWeightTracks
      this.inputOffset = track.aclTrackOffset
      this.isBlendWeights = true
    }
    else {
      this.compressedTracks = clip.aclTransforms.compressed
      this.decompressedTracks = clip.aclTransforms.decompressed

      let numOutputValues = 0
      let inputOffset = 0
      if (track.propertyName === 'quaternion') {
        numOutputValues = 4
        inputOffset = 0
      }
      else if (track.propertyName === 'position') {
        numOutputValues = 3
        inputOffset = 4
      }
      else if (track.propertyName === 'scale') {
        numOutputValues = 3
        inputOffset = 8
      }

      this.numOutputValues = numOutputValues
      this.inputOffset = track.aclTrackIndex * 12 + inputOffset // 12 floats per QVV
      this.aclTrackIndex = track.aclTrackIndex
      this.isBlendWeights = false
    }

    this.decoder = decoder
    this.resultBuffer = result
  }

  evaluate(t) {
    if (this.isBlendWeights) {
      const inputOffset = this.inputOffset
      for (let i = 0; i < this.numOutputValues; ++i) {
        this.decoder.decompressTrack(this.compressedTracks, inputOffset + i, t, RoundingPolicy.None, this.decompressedTracks)

        const array = this.decompressedTracks.array
        this.resultBuffer[i] = array[inputOffset + i]
      }
    }
    else {
      this.decoder.decompressTrack(this.compressedTracks, this.aclTrackIndex, t, RoundingPolicy.None, this.decompressedTracks)

      const array = this.decompressedTracks.array
      const inputOffset = this.inputOffset
      for (let i = 0; i < this.numOutputValues; ++i) {
        this.resultBuffer[i] = array[inputOffset + i]
      }
    }

    return this.resultBuffer
  }
}
// nfrechette - END

export class Viewer {

  constructor (el, options) {
    this.el = el;
    this.options = options;

    this.lights = [];
    this.content = null;
    this.mixer = null;
    this.clips = [];
    this.gui = null;

    this.state = {
      environment: options.preset === Preset.ASSET_GENERATOR
        ? environments.find((e) => e.id === 'footprint-court').name
        : environments[1].name,
      background: false,
      playbackSpeed: 1.0,
      actionStates: {},
      camera: DEFAULT_CAMERA,
      wireframe: false,
      skeleton: false,
      grid: false,

      // nfrechette - BEGIN
      animSource: 'ACL Compressed',
      jointPrecision: 0.0001, // 0.01cm
      jointShellDistance: 1.0, // 1.0m
      blendWeightPrecision: 0.0001,
      // nfrechette - END

      // Lights
      addLights: true,
      exposure: 1.0,
      textureEncoding: 'sRGB',
      ambientIntensity: 0.3,
      ambientColor: 0xFFFFFF,
      directIntensity: 0.8 * Math.PI, // TODO(#116)
      directColor: 0xFFFFFF,
      bgColor1: '#ffffff',
      bgColor2: '#353535'
    };

    this.prevTime = 0;

    this.stats = new Stats();
    this.stats.dom.height = '48px';
    [].forEach.call(this.stats.dom.children, (child) => (child.style.display = ''));

    this.scene = new Scene();

    const fov = options.preset === Preset.ASSET_GENERATOR
      ? 0.8 * 180 / Math.PI
      : 60;
    this.defaultCamera = new PerspectiveCamera( fov, el.clientWidth / el.clientHeight, 0.01, 1000 );
    this.activeCamera = this.defaultCamera;
    this.scene.add( this.defaultCamera );

    this.renderer = window.renderer = new WebGLRenderer({antialias: true});
    this.renderer.physicallyCorrectLights = true;
    this.renderer.outputEncoding = sRGBEncoding;
    this.renderer.setClearColor( 0xcccccc );
    this.renderer.setPixelRatio( window.devicePixelRatio );
    this.renderer.setSize( el.clientWidth, el.clientHeight );

    this.pmremGenerator = new PMREMGenerator( this.renderer );
    this.pmremGenerator.compileEquirectangularShader();

    this.controls = new OrbitControls( this.defaultCamera, this.renderer.domElement );
    this.controls.autoRotate = false;
    this.controls.autoRotateSpeed = -10;
    this.controls.screenSpacePanning = true;

    this.vignette = createBackground({
      aspect: this.defaultCamera.aspect,
      grainScale: IS_IOS ? 0 : 0.001, // mattdesl/three-vignette-background#1
      colors: [this.state.bgColor1, this.state.bgColor2]
    });
    this.vignette.name = 'Vignette';
    this.vignette.renderOrder = -1;

    this.el.appendChild(this.renderer.domElement);

    this.cameraCtrl = null;
    this.cameraFolder = null;
    this.animFolder = null;
    this.animCtrls = [];
    this.morphFolder = null;
    this.morphCtrls = [];
    this.skeletonHelpers = [];
    this.gridHelper = null;
    this.axesHelper = null;

    this.addAxesHelper();
    this.addGUI();
    if (options.kiosk) this.gui.close();

    this.animate = this.animate.bind(this);
    requestAnimationFrame( this.animate );
    window.addEventListener('resize', this.resize.bind(this), false);

    // nfrechette - BEGIN
    this._aclEncoder = new Encoder()
    this._aclDecoder = new Decoder()
    // nfrechette - END
  }

  animate (time) {

    requestAnimationFrame( this.animate );

    const dt = (time - this.prevTime) / 1000;

    this.controls.update();
    this.stats.update();
    this.mixer && this.mixer.update(dt);
    this.render();

    this.prevTime = time;

  }

  render () {

    this.renderer.render( this.scene, this.activeCamera );
    if (this.state.grid) {
      this.axesCamera.position.copy(this.defaultCamera.position)
      this.axesCamera.lookAt(this.axesScene.position)
      this.axesRenderer.render( this.axesScene, this.axesCamera );
    }
  }

  resize () {

    const {clientHeight, clientWidth} = this.el.parentElement;

    this.defaultCamera.aspect = clientWidth / clientHeight;
    this.defaultCamera.updateProjectionMatrix();
    this.vignette.style({aspect: this.defaultCamera.aspect});
    this.renderer.setSize(clientWidth, clientHeight);

    this.axesCamera.aspect = this.axesDiv.clientWidth / this.axesDiv.clientHeight;
    this.axesCamera.updateProjectionMatrix();
    this.axesRenderer.setSize(this.axesDiv.clientWidth, this.axesDiv.clientHeight);
  }

  load ( url, rootPath, assetMap ) {

    const baseURL = LoaderUtils.extractUrlBase(url);

    // Load.
    return new Promise((resolve, reject) => {

      const manager = new LoadingManager();

      // Intercept and override relative URLs.
      manager.setURLModifier((url, path) => {

        // URIs in a glTF file may be escaped, or not. Assume that assetMap is
        // from an un-escaped source, and decode all URIs before lookups.
        // See: https://github.com/donmccurdy/three-gltf-viewer/issues/146
        const normalizedURL = rootPath + decodeURI(url)
          .replace(baseURL, '')
          .replace(/^(\.?\/)/, '');

        if (assetMap.has(normalizedURL)) {
          const blob = assetMap.get(normalizedURL);
          const blobURL = URL.createObjectURL(blob);
          blobURLs.push(blobURL);
          return blobURL;
        }

        return (path || '') + url;

      });

      const loader = new GLTFLoader(manager);
      loader.setCrossOrigin('anonymous');

      const dracoLoader = new DRACOLoader();
      dracoLoader.setDecoderPath( 'assets/draco/' );
      loader.setDRACOLoader( dracoLoader );

      const blobURLs = [];

      loader.load(url, (gltf) => {

        const scene = gltf.scene || gltf.scenes[0];
        const clips = gltf.animations || [];

        if (!scene) {
          // Valid, but not supported by this viewer.
          throw new Error(
            'This model contains no scene, and cannot be viewed here. However,'
            + ' it may contain individual 3D resources.'
          );
        }

        this.setContent(scene, clips);

        blobURLs.forEach(URL.revokeObjectURL);

        // See: https://github.com/google/draco/issues/349
        // DRACOLoader.releaseDecoderModule();

        resolve(gltf);

      }, undefined, reject);

    });

  }

  /**
   * @param {THREE.Object3D} object
   * @param {Array<THREE.AnimationClip} clips
   */
  setContent ( object, clips ) {

    this.clear();

    const box = new Box3().setFromObject(object);
    const size = box.getSize(new Vector3()).length();
    const center = box.getCenter(new Vector3());

    this.controls.reset();

    object.position.x += (object.position.x - center.x);
    object.position.y += (object.position.y - center.y);
    object.position.z += (object.position.z - center.z);
    this.controls.maxDistance = size * 10;
    this.defaultCamera.near = size / 100;
    this.defaultCamera.far = size * 100;
    this.defaultCamera.updateProjectionMatrix();

    if (this.options.cameraPosition) {

      this.defaultCamera.position.fromArray( this.options.cameraPosition );
      this.defaultCamera.lookAt( new Vector3() );

    } else {

      this.defaultCamera.position.copy(center);
      this.defaultCamera.position.x += size / 2.0;
      this.defaultCamera.position.y += size / 5.0;
      this.defaultCamera.position.z += size / 2.0;
      this.defaultCamera.lookAt(center);

    }

    this.setCamera(DEFAULT_CAMERA);

    this.axesCamera.position.copy(this.defaultCamera.position)
    this.axesCamera.lookAt(this.axesScene.position)
    this.axesCamera.near = size / 100;
    this.axesCamera.far = size * 100;
    this.axesCamera.updateProjectionMatrix();
    this.axesCorner.scale.set(size, size, size);

    this.controls.saveState();

    this.scene.add(object);
    this.content = object;

    this.state.addLights = true;

    this.content.traverse((node) => {
      if (node.isLight) {
        this.state.addLights = false;
      } else if (node.isMesh) {
        // TODO(https://github.com/mrdoob/three.js/pull/18235): Clean up.
        node.material.depthWrite = !node.material.transparent;
      }
    });

    this.setClips(clips);

    this.updateLights();
    this.updateGUI();
    this.updateEnvironment();
    this.updateTextureEncoding();
    this.updateDisplay();

    window.content = this.content;
    console.info('[glTF Viewer] THREE.Scene exported as `window.content`.');
    this.printGraph(this.content);

  }

  printGraph (node) {
    return // nfrechette
    console.group(' <' + node.type + '> ' + node.name);
    node.children.forEach((child) => this.printGraph(child));
    console.groupEnd();

  }

  /**
   * @param {Array<THREE.AnimationClip} clips
   */
  setClips ( clips ) {
    if (this.mixer) {
      this.mixer.stopAllAction();
      this.mixer.uncacheRoot(this.mixer.getRoot());
      this.mixer = null;
    }

    this.freeACLMemory(this.clips)  // nfrechette

    this.clips = clips;
    if (!clips.length) return;

    this.mixer = new AnimationMixer( this.content );

    this.compressWithACL(clips) // nfrechette
  }

  // nfrechette - BEGIN
  findClipDuration ( clip ) {
    const numClipTracks = clip.tracks.length
    let maxDuration = 0.0
    let maxNumSamples = 0
    for (let trackIndex = 0; trackIndex < numClipTracks; ++trackIndex) {
      const timeValues = clip.tracks[trackIndex].times
      const numSamples = timeValues.length
      if (numSamples > 0) {
        const lastSampleTime = timeValues[numSamples - 1]
        maxDuration = Math.max(maxDuration, lastSampleTime)
        maxNumSamples = Math.max(maxNumSamples, numSamples)
      }
    }

    if (maxNumSamples === 0) {
      return 0.0  // No samples means we have no duration
    }

    if (maxNumSamples === 1) {
      return Number.POSITIVE_INFINITY // A single sample means we have an indefinite duration (static pose)
    }

    // Use the value of our last sample time
    return maxDuration
  }

  calcNumSamples ( duration, sampleRate ) {
    if (duration === 0.0) {
      return 0  // No duration whatsoever, we have no samples
    }

    if (duration === Number.POSITIVE_INFINITY) {
      return 1  // An infinite duration, we have a single sample (static pose)
    }

    // Otherwise we have at least 1 sample
    return Math.floor((duration * sampleRate) + 0.5) + 1
  }

  findClipSampleRate ( clip, duration ) {
    // If our clip has no samples, or a single sample, or samples which aren't spaced linearly,
    // we will try and use a sample rate as close to 30 FPS as possible.
    if (duration === 0.0 || duration === Number.POSITIVE_INFINITY) {
      return 30.0
    }

    // First find the smallest amount of time between two samples, this is potentially
    // our sample rate.
    const numClipTracks = clip.tracks.length
    let smallestDeltaTime = 1.0E10;
    for (let trackIndex = 0; trackIndex < numClipTracks; ++trackIndex) {
      const timeValues = clip.tracks[trackIndex].times
      const numSamples = timeValues.length
      for (let sampleIndex = 1; sampleIndex < numSamples; ++sampleIndex) {
        const prevSampleTime = timeValues[sampleIndex - 1];
        const currSampleTime = timeValues[sampleIndex];
        const deltaTime = currSampleTime - prevSampleTime;
        smallestDeltaTime = Math.min(smallestDeltaTime, deltaTime);
      }
    }

    if (!Number.isFinite(smallestDeltaTime) || smallestDeltaTime < 0.0) {
      throw new RangeError('Invalid data')
    }

    // If every sample is near a multiple of our sample rate, it is valid
    let isSampleRateValid = true
    for (let trackIndex = 0; trackIndex < numClipTracks; ++trackIndex) {
      const timeValues = clip.tracks[trackIndex].times
      const numSamples = timeValues.length
      for (let sampleIndex = 0; sampleIndex < numSamples; ++sampleIndex) {
        const sampleTime = timeValues[sampleIndex];
        const remainder = sampleTime % smallestDeltaTime
        if (remainder > 0.001) {
          isSampleRateValid = false
          break
        }
      }
    }

    if (isSampleRateValid) {
      return 1.0 / smallestDeltaTime
    }

    // Our samples aren't linearly spaced, do our best to approximate something decent
    const numSamples = this.calcNumSamples(duration, 30.0)
    return (numSamples - 1.0) / duration
  }

  convertClipToACLTracks ( clip ) {
    const interpolantSettings = {
      endingStart: ZeroCurvatureEnding,
      endingEnd: ZeroCurvatureEnding
    };

    let resultBuffer = new Float64Array(4)

    // Find all nodes
    const nodes = {}
    this.content.traverse(node => nodes[node.uuid] = node)
    const nodeUUIDs = Object.keys(nodes)
    const numTracks = nodeUUIDs.length
    let numBlendWeightTracks = 0
    const blendWeightProps = []

    // Find the props that have data
    const propsWithData = {}
    for (let trackIndex = 0; trackIndex < clip.tracks.length; ++trackIndex) {
      const track = clip.tracks[trackIndex]
      const parsedPath = false
      const prop = PropertyBinding.create(this.content, track.name, parsedPath)
      prop.track = track
      prop.track.propertyName = prop.parsedPath.propertyName

      if (prop.parsedPath.propertyName === 'morphTargetInfluences') {
        blendWeightProps.push(prop)
        const numBlendWeights = prop.node.morphTargetInfluences.length
        track.numBlendWeightTracks = numBlendWeights

        track.aclTrackOffset = numBlendWeightTracks
        numBlendWeightTracks += numBlendWeights
      }

      if (!propsWithData[prop.node.uuid]) {
        propsWithData[prop.node.uuid] = []
      }
      propsWithData[prop.node.uuid].push(prop)
    }

    const duration = this.findClipDuration(clip)
    const sampleRate = this.findClipSampleRate(clip, duration)
    const numSamplesPerTrack = this.calcNumSamples(duration, sampleRate)

    const qvvTracks = new TrackArray(numTracks, SampleTypes.QVV, numSamplesPerTrack, sampleRate);
    const weightTracks = new TrackArray(numBlendWeightTracks, SampleTypes.Float, numSamplesPerTrack, sampleRate);

    // TODO: Display this in the viewport?
    console.log(`num transform tracks: ${qvvTracks.numTracks}`)
    console.log(`num blend weight tracks: ${weightTracks.numTracks}`)
    console.log(`sample rate: ${qvvTracks.sampleRate}`)
    console.log(`num samples per track: ${qvvTracks.numSamplesPerTrack}`)
    console.log(`duration: ${qvvTracks.duration}`)

    // Populate our QVV track data
    for (let trackIndex = 0; trackIndex < numTracks; ++trackIndex) {
      const nodeUUID = nodeUUIDs[trackIndex]
      const node = nodes[nodeUUID]

      const qvvTrack = qvvTracks.at(trackIndex)

      // Setup our description
      if (node.parent) {
        qvvTrack.description.parentIndex = nodeUUIDs.indexOf(node.parent.uuid)
        qvvTrack.description.precision = this.state.jointPrecision
        qvvTrack.description.shellDistance = this.state.jointShellDistance
      }

      const qvv = QVV.identity
      if (node.quaternion) {
        qvv.rotation.x = node.quaternion.x
        qvv.rotation.y = node.quaternion.y
        qvv.rotation.z = node.quaternion.z
        qvv.rotation.w = node.quaternion.w
        qvv.rotation.normalize()
      }
      if (node.translation) {
        qvv.translation.x = node.translation.x
        qvv.translation.y = node.translation.y
        qvv.translation.z = node.translation.z
      }
      if (node.scale) {
        qvv.scale.x = node.scale.x
        qvv.scale.y = node.scale.y
        qvv.scale.z = node.scale.z
      }

      // Write the bind pose to every sample
      for (let sampleIndex = 0; sampleIndex < numSamplesPerTrack; ++sampleIndex) {
        const sample = qvvTrack.at(sampleIndex)
        sample.setQVV(qvv)
      }

      const props = propsWithData[nodeUUID]
      const numAnimatedProps = props ? props.length : 0
      for (let propIndex = 0; propIndex < numAnimatedProps; ++propIndex) {
        const prop = props[propIndex]
        const propertyName = prop.parsedPath.propertyName

        // Add metadata
        prop.track.aclTrackIndex = trackIndex

        // This track is animated, read the data
        const interpolant = prop.track.createInterpolant(null)
        interpolant.settings = interpolantSettings
        interpolant.resultBuffer = resultBuffer

        if (propertyName === 'quaternion') {
          for (let sampleIndex = 0; sampleIndex < numSamplesPerTrack; ++sampleIndex) {
            const sampleTime = Math.min(sampleIndex / sampleRate, duration)
            interpolant.evaluate(sampleTime)

            const sample = qvvTrack.at(sampleIndex)
            sample.getQVV(qvv)

            qvv.rotation.x = resultBuffer[0]
            qvv.rotation.y = resultBuffer[1]
            qvv.rotation.z = resultBuffer[2]
            qvv.rotation.w = resultBuffer[3]
            qvv.rotation.normalize()

            sample.setQVV(qvv)
          }
        }
        else if (propertyName === 'position') {
          for (let sampleIndex = 0; sampleIndex < numSamplesPerTrack; ++sampleIndex) {
            const sampleTime = Math.min(sampleIndex / sampleRate, duration)
            interpolant.evaluate(sampleTime)

            const sample = qvvTrack.at(sampleIndex)
            sample.getQVV(qvv)

            qvv.translation.x = resultBuffer[0]
            qvv.translation.y = resultBuffer[1]
            qvv.translation.z = resultBuffer[2]

            sample.setQVV(qvv)
          }
        }
        else if (propertyName === 'scale') {
          for (let sampleIndex = 0; sampleIndex < numSamplesPerTrack; ++sampleIndex) {
            const sampleTime = Math.min(sampleIndex / sampleRate, duration)
            interpolant.evaluate(sampleTime)

            const sample = qvvTrack.at(sampleIndex)
            sample.getQVV(qvv)

            qvv.scale.x = resultBuffer[0]
            qvv.scale.y = resultBuffer[1]
            qvv.scale.z = resultBuffer[2]

            sample.setQVV(qvv)
          }
        }
      }
    }

    // Populate our blend weight track data
    blendWeightProps.forEach((prop) => {
      const numPropTracks = prop.track.numBlendWeightTracks
      const trackOffset = prop.track.aclTrackOffset

      // Set our track settings
      for (let trackIndex = 0; trackIndex < numPropTracks; ++trackIndex) {
        const weightTrack = weightTracks.at(trackOffset + trackIndex)
        weightTrack.description.precision = this.state.blendWeightPrecision
      }

      // Make sure our buffer is large enough
      if (resultBuffer.length < numPropTracks) {
        resultBuffer = new Float64Array(numPropTracks)
      }

      // This track is animated, read the data
      const interpolant = prop.track.createInterpolant(null)
      interpolant.settings = interpolantSettings
      interpolant.resultBuffer = resultBuffer

      for (let sampleIndex = 0; sampleIndex < numSamplesPerTrack; ++sampleIndex) {
        const sampleTime = Math.min(sampleIndex / sampleRate, duration)
        interpolant.evaluate(sampleTime)

        for (let trackIndex = 0; trackIndex < numPropTracks; ++trackIndex) {
          const weightTrack = weightTracks.at(trackOffset + trackIndex)

          const sample = weightTrack.at(sampleIndex)
          sample.setFloat(resultBuffer[trackIndex])
        }
      }
    })

    return { transforms: qvvTracks, weights: weightTracks }
  }

  bindACLProxy ( clip ) {
    const decoder = this._aclDecoder

    clip.tracks.forEach((track) => {
      track.createInterpolantRef = track.createInterpolant
      track.createInterpolantACLRaw = function (result) {
        return new ACLRawInterpolant(clip, track, result)
      }
      track.createInterpolantACL = function (result) {
        return new ACLInterpolant(clip, decoder, track, result)
      }
      track.createInterpolantACLPerTrack = function (result) {
        return new ACLPerTrackInterpolant(clip, decoder, track, result)
      }
    })
  }

  freeACLMemory ( clips ) {
    if (clips) {
      clips.forEach((clip) => {
        if (clip.aclTransforms) {
          clip.aclTransforms.compressed.dispose()
          clip.aclTransforms.decompressed.dispose()
          clip.aclTransforms = null
        }

        if (clip.aclWeights) {
          clip.aclWeights.compressed.dispose()
          clip.aclWeights.decompressed.dispose()
          clip.aclWeights = null
        }

        if (clip.refTracks) {
          clip.tracks = clip.refTracks
          clip.refTracks = null
          clip.aclTracks = null
        }

        clip.tracks.forEach((track) => {
          if (track.createInterpolantRef) {
            track.createInterpolant = track.createInterpolantRef
            track.createInterpolantRef = null
            track.createInterpolantACLRaw = null
            track.createInterpolantACL = null
            track.createInterpolantACLPerTrack = null
          }
        })
      })
    }
  }

  compressWithACL ( clips ) {
    const aclPromise = Promise.all([this._aclEncoder.isReady(), this._aclDecoder.isReady()])

    clips.forEach((clip) => {
      // Cache our original tracks and convert them to ACL tracks
      clip.refTracks = clip.tracks
      clip.aclTracks = this.convertClipToACLTracks(clip)

      aclPromise.then(() => {
        // Compress our clip with ACL
        if (clip.aclTracks.transforms.numTracks > 0) {
          const aclTransforms = { raw: clip.aclTracks.transforms }

          const compressedTracks = this._aclEncoder.compress(aclTransforms.raw)
          compressedTracks.bind(this._aclDecoder)

          console.log(`ACL transform compressed size: ${compressedTracks.byteLength} bytes`)

          aclTransforms.compressed = compressedTracks
          aclTransforms.decompressed = new DecompressedTracks(this._aclDecoder)

          clip.aclTransforms = aclTransforms
        }

        if (clip.aclTracks.weights.numTracks > 0) {
          const aclWeights = { raw: clip.aclTracks.weights }

          const compressedTracks = this._aclEncoder.compress(aclWeights.raw)
          compressedTracks.bind(this._aclDecoder)

          console.log(`ACL weights compressed size: ${compressedTracks.byteLength} bytes`)

          aclWeights.compressed = compressedTracks
          aclWeights.decompressed = new DecompressedTracks(this._aclDecoder)

          clip.aclWeights = aclWeights
        }

        // Bind our clip to playback from ACL tracks
        this.bindACLProxy(clip)
      })
    })

    aclPromise.then(() => {
      this.updateAnimationSource()
    })
  }

  updateAnimationSource () {
    console.log(`Changing animation source to: ${this.state.animSource}`)

    // Replace the track interpolant with the one we need
    this.clips.forEach((clip) => {
      clip.tracks.forEach((track) => {
        if (this.state.animSource === 'Source File') {
          track.createInterpolant = track.createInterpolantRef
        }
        else if (this.state.animSource === 'ACL Raw') {
          track.createInterpolant = track.createInterpolantACLRaw
        }
        else if (this.state.animSource === 'ACL Compressed') {
          track.createInterpolant = track.createInterpolantACL
        }
        else if (this.state.animSource === 'ACL Compressed (per track)') {
          track.createInterpolant = track.createInterpolantACLPerTrack
        }
      })

      // Always reset our ACL decompressed tracks cache
      if (clip.aclTransforms) {
        clip.aclTransforms.decompressed.cachedTime = -1.0
      }
      if (clip.aclWeights) {
        clip.aclWeights.decompressed.cachedTime = -1.0
      }

      // Remove any stale data from the mixer
      this.mixer.uncacheClip(clip)

      // Restart playback if we were playing
      if (this.state.actionStates[clip.name]) {
        const action = this.mixer.clipAction(clip);
        action.play();
      }

      // Measure perf
      const dt = 1.0 / 30.0
      console.time('playback')
      for (let i = 0; i < 1000; ++i) {
        this.mixer.update(dt)
      }
      console.timeEnd('playback')
      this.mixer.setTime(0.0)
    })
  }

  animationCompressionSettingChanged () {
    // Our compression settings have changed, just reset our clips
    this.setClips(this.clips)
  }
  // nfrechette - END

  playAllClips () {
    this.clips.forEach((clip) => {
      this.mixer.clipAction(clip).reset().play();
      this.state.actionStates[clip.name] = true;
    });
  }

  /**
   * @param {string} name
   */
  setCamera ( name ) {
    if (name === DEFAULT_CAMERA) {
      this.controls.enabled = true;
      this.activeCamera = this.defaultCamera;
    } else {
      this.controls.enabled = false;
      this.content.traverse((node) => {
        if (node.isCamera && node.name === name) {
          this.activeCamera = node;
        }
      });
    }
  }

  updateTextureEncoding () {
    const encoding = this.state.textureEncoding === 'sRGB'
      ? sRGBEncoding
      : LinearEncoding;
    traverseMaterials(this.content, (material) => {
      if (material.map) material.map.encoding = encoding;
      if (material.emissiveMap) material.emissiveMap.encoding = encoding;
      if (material.map || material.emissiveMap) material.needsUpdate = true;
    });
  }

  updateLights () {
    const state = this.state;
    const lights = this.lights;

    if (state.addLights && !lights.length) {
      this.addLights();
    } else if (!state.addLights && lights.length) {
      this.removeLights();
    }

    this.renderer.toneMappingExposure = state.exposure;

    if (lights.length === 2) {
      lights[0].intensity = state.ambientIntensity;
      lights[0].color.setHex(state.ambientColor);
      lights[1].intensity = state.directIntensity;
      lights[1].color.setHex(state.directColor);
    }
  }

  addLights () {
    const state = this.state;

    if (this.options.preset === Preset.ASSET_GENERATOR) {
      const hemiLight = new HemisphereLight();
      hemiLight.name = 'hemi_light';
      this.scene.add(hemiLight);
      this.lights.push(hemiLight);
      return;
    }

    const light1  = new AmbientLight(state.ambientColor, state.ambientIntensity);
    light1.name = 'ambient_light';
    this.defaultCamera.add( light1 );

    const light2  = new DirectionalLight(state.directColor, state.directIntensity);
    light2.position.set(0.5, 0, 0.866); // ~60º
    light2.name = 'main_light';
    this.defaultCamera.add( light2 );

    this.lights.push(light1, light2);
  }

  removeLights () {

    this.lights.forEach((light) => light.parent.remove(light));
    this.lights.length = 0;

  }

  updateEnvironment () {

    const environment = environments.filter((entry) => entry.name === this.state.environment)[0];

    this.getCubeMapTexture( environment ).then(( { envMap } ) => {

      if ((!envMap || !this.state.background) && this.activeCamera === this.defaultCamera) {
        this.scene.add(this.vignette);
      } else {
        this.scene.remove(this.vignette);
      }

      this.scene.environment = envMap;
      this.scene.background = this.state.background ? envMap : null;

    });

  }

  getCubeMapTexture ( environment ) {
    const { path } = environment;

    // no envmap
    if ( ! path ) return Promise.resolve( { envMap: null } );

    return new Promise( ( resolve, reject ) => {

      new RGBELoader()
        .setDataType( UnsignedByteType )
        .load( path, ( texture ) => {

          const envMap = this.pmremGenerator.fromEquirectangular( texture ).texture;
          this.pmremGenerator.dispose();

          resolve( { envMap } );

        }, undefined, reject );

    });

  }

  updateDisplay () {
    if (this.skeletonHelpers.length) {
      this.skeletonHelpers.forEach((helper) => this.scene.remove(helper));
    }

    traverseMaterials(this.content, (material) => {
      material.wireframe = this.state.wireframe;
    });

    this.content.traverse((node) => {
      if (node.isMesh && node.skeleton && this.state.skeleton) {
        const helper = new SkeletonHelper(node.skeleton.bones[0].parent);
        helper.material.linewidth = 3;
        this.scene.add(helper);
        this.skeletonHelpers.push(helper);
      }
    });

    if (this.state.grid !== Boolean(this.gridHelper)) {
      if (this.state.grid) {
        this.gridHelper = new GridHelper();
        this.axesHelper = new AxesHelper();
        this.axesHelper.renderOrder = 999;
        this.axesHelper.onBeforeRender = (renderer) => renderer.clearDepth();
        this.scene.add(this.gridHelper);
        this.scene.add(this.axesHelper);
      } else {
        this.scene.remove(this.gridHelper);
        this.scene.remove(this.axesHelper);
        this.gridHelper = null;
        this.axesHelper = null;
        this.axesRenderer.clear();
      }
    }
  }

  updateBackground () {
    this.vignette.style({colors: [this.state.bgColor1, this.state.bgColor2]});
  }

  /**
   * Adds AxesHelper.
   *
   * See: https://stackoverflow.com/q/16226693/1314762
   */
  addAxesHelper () {
    this.axesDiv = document.createElement('div');
    this.el.appendChild( this.axesDiv );
    this.axesDiv.classList.add('axes');

    const {clientWidth, clientHeight} = this.axesDiv;

    this.axesScene = new Scene();
    this.axesCamera = new PerspectiveCamera( 50, clientWidth / clientHeight, 0.1, 10 );
    this.axesScene.add( this.axesCamera );

    this.axesRenderer = new WebGLRenderer( { alpha: true } );
    this.axesRenderer.setPixelRatio( window.devicePixelRatio );
    this.axesRenderer.setSize( this.axesDiv.clientWidth, this.axesDiv.clientHeight );

    this.axesCamera.up = this.defaultCamera.up;

    this.axesCorner = new AxesHelper(5);
    this.axesScene.add( this.axesCorner );
    this.axesDiv.appendChild(this.axesRenderer.domElement);
  }

  addGUI () {

    const gui = this.gui = new GUI({autoPlace: false, width: 260, hideable: true});

    // Display controls.
    const dispFolder = gui.addFolder('Display');
    const envBackgroundCtrl = dispFolder.add(this.state, 'background');
    envBackgroundCtrl.onChange(() => this.updateEnvironment());
    const wireframeCtrl = dispFolder.add(this.state, 'wireframe');
    wireframeCtrl.onChange(() => this.updateDisplay());
    const skeletonCtrl = dispFolder.add(this.state, 'skeleton');
    skeletonCtrl.onChange(() => this.updateDisplay());
    const gridCtrl = dispFolder.add(this.state, 'grid');
    gridCtrl.onChange(() => this.updateDisplay());
    dispFolder.add(this.controls, 'autoRotate');
    dispFolder.add(this.controls, 'screenSpacePanning');
    const bgColor1Ctrl = dispFolder.addColor(this.state, 'bgColor1');
    const bgColor2Ctrl = dispFolder.addColor(this.state, 'bgColor2');
    bgColor1Ctrl.onChange(() => this.updateBackground());
    bgColor2Ctrl.onChange(() => this.updateBackground());

    // Lighting controls.
    const lightFolder = gui.addFolder('Lighting');
    const encodingCtrl = lightFolder.add(this.state, 'textureEncoding', ['sRGB', 'Linear']);
    encodingCtrl.onChange(() => this.updateTextureEncoding());
    lightFolder.add(this.renderer, 'outputEncoding', {sRGB: sRGBEncoding, Linear: LinearEncoding})
      .onChange(() => {
        this.renderer.outputEncoding = Number(this.renderer.outputEncoding);
        traverseMaterials(this.content, (material) => {
          material.needsUpdate = true;
        });
      });
    const envMapCtrl = lightFolder.add(this.state, 'environment', environments.map((env) => env.name));
    envMapCtrl.onChange(() => this.updateEnvironment());
    [
      lightFolder.add(this.state, 'exposure', 0, 2),
      lightFolder.add(this.state, 'addLights').listen(),
      lightFolder.add(this.state, 'ambientIntensity', 0, 2),
      lightFolder.addColor(this.state, 'ambientColor'),
      lightFolder.add(this.state, 'directIntensity', 0, 4), // TODO(#116)
      lightFolder.addColor(this.state, 'directColor')
    ].forEach((ctrl) => ctrl.onChange(() => this.updateLights()));

    // Animation controls.
    this.animFolder = gui.addFolder('Animation');
    this.animFolder.domElement.style.display = 'none';
    const playbackSpeedCtrl = this.animFolder.add(this.state, 'playbackSpeed', 0, 1);
    playbackSpeedCtrl.onChange((speed) => {
      if (this.mixer) this.mixer.timeScale = speed;
    });
    this.animFolder.add({playAll: () => this.playAllClips()}, 'playAll');

    // nfrechette - BEGIN
    const animSourceCtrl = this.animFolder.add(this.state, 'animSource', ['Source File', 'ACL Raw', 'ACL Compressed', 'ACL Compressed (per track)'])
    animSourceCtrl.onChange(() => this.updateAnimationSource())
    const jointPrecisionCtrl = this.animFolder.add(this.state, 'jointPrecision')
    jointPrecisionCtrl.onChange(() => this.animationCompressionSettingChanged())
    const jointShellDistanceCtrl = this.animFolder.add(this.state, 'jointShellDistance')
    jointShellDistanceCtrl.onChange(() => this.animationCompressionSettingChanged())
    const blendWeightPrecisionCtrl = this.animFolder.add(this.state, 'blendWeightPrecision')
    blendWeightPrecisionCtrl.onChange(() => this.animationCompressionSettingChanged())
    // nfrechette - END

    // Morph target controls.
    this.morphFolder = gui.addFolder('Morph Targets');
    this.morphFolder.domElement.style.display = 'none';

    // Camera controls.
    this.cameraFolder = gui.addFolder('Cameras');
    this.cameraFolder.domElement.style.display = 'none';

    // Stats.
    const perfFolder = gui.addFolder('Performance');
    const perfLi = document.createElement('li');
    this.stats.dom.style.position = 'static';
    perfLi.appendChild(this.stats.dom);
    perfLi.classList.add('gui-stats');
    perfFolder.__ul.appendChild( perfLi );

    const guiWrap = document.createElement('div');
    this.el.appendChild( guiWrap );
    guiWrap.classList.add('gui-wrap');
    guiWrap.appendChild(gui.domElement);
    gui.open();

  }

  updateGUI () {
    this.cameraFolder.domElement.style.display = 'none';

    this.morphCtrls.forEach((ctrl) => ctrl.remove());
    this.morphCtrls.length = 0;
    this.morphFolder.domElement.style.display = 'none';

    this.animCtrls.forEach((ctrl) => ctrl.remove());
    this.animCtrls.length = 0;
    this.animFolder.domElement.style.display = 'none';

    const cameraNames = [];
    const morphMeshes = [];
    this.content.traverse((node) => {
      if (node.isMesh && node.morphTargetInfluences) {
        morphMeshes.push(node);
      }
      if (node.isCamera) {
        node.name = node.name || `VIEWER__camera_${cameraNames.length + 1}`;
        cameraNames.push(node.name);
      }
    });

    if (cameraNames.length) {
      this.cameraFolder.domElement.style.display = '';
      if (this.cameraCtrl) this.cameraCtrl.remove();
      const cameraOptions = [DEFAULT_CAMERA].concat(cameraNames);
      this.cameraCtrl = this.cameraFolder.add(this.state, 'camera', cameraOptions);
      this.cameraCtrl.onChange((name) => this.setCamera(name));
    }

    if (morphMeshes.length) {
      this.morphFolder.domElement.style.display = '';
      morphMeshes.forEach((mesh) => {
        if (mesh.morphTargetInfluences.length) {
          const nameCtrl = this.morphFolder.add({name: mesh.name || 'Untitled'}, 'name');
          this.morphCtrls.push(nameCtrl);
        }
        for (let i = 0; i < mesh.morphTargetInfluences.length; i++) {
          const ctrl = this.morphFolder.add(mesh.morphTargetInfluences, i, 0, 1, 0.01).listen();
          Object.keys(mesh.morphTargetDictionary).forEach((key) => {
            if (key && mesh.morphTargetDictionary[key] === i) ctrl.name(key);
          });
          this.morphCtrls.push(ctrl);
        }
      });
    }

    if (this.clips.length) {
      this.animFolder.domElement.style.display = '';
      const actionStates = this.state.actionStates = {};
      this.clips.forEach((clip, clipIndex) => {
        // Autoplay the first clip.
        let action;
        if (clipIndex === 0) {
          actionStates[clip.name] = true;
          action = this.mixer.clipAction(clip);
          action.play();
        } else {
          actionStates[clip.name] = false;
        }

        // Play other clips when enabled.
        const ctrl = this.animFolder.add(actionStates, clip.name).listen();
        ctrl.onChange((playAnimation) => {
          action = action || this.mixer.clipAction(clip);
          action.setEffectiveTimeScale(1);
          playAnimation ? action.play() : action.stop();
        });
        this.animCtrls.push(ctrl);
      });
    }
  }

  clear () {

    if ( !this.content ) return;

    this.scene.remove( this.content );

    // dispose geometry
    this.content.traverse((node) => {

      if ( !node.isMesh ) return;

      node.geometry.dispose();

    } );

    // dispose textures
    traverseMaterials( this.content, (material) => {

      MAP_NAMES.forEach( (map) => {

        if (material[ map ]) material[ map ].dispose();

      } );

    } );

  }

};

function traverseMaterials (object, callback) {
  object.traverse((node) => {
    if (!node.isMesh) return;
    const materials = Array.isArray(node.material)
      ? node.material
      : [node.material];
    materials.forEach(callback);
  });
}
