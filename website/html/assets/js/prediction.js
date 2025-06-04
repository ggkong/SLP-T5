document.addEventListener('DOMContentLoaded', function() {
    // 文件上传处理
    const fileDropZone = document.querySelector('.file-drop-zone');
    const fileInput = document.querySelector('.file-input');
    const filePreview = document.querySelector('.file-preview');
    const fileName = document.querySelector('.file-name');
    const fileSize = document.querySelector('.file-size');
    const dropZoneContent = document.querySelector('.drop-zone-content');
    const removeFileBtn = document.querySelector('.remove-file');

    // 拖拽效果
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        fileDropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        fileDropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        fileDropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        fileDropZone.classList.add('drag-over');
    }

    function unhighlight(e) {
        fileDropZone.classList.remove('drag-over');
    }

    // 处理文件上传
    fileDropZone.addEventListener('drop', handleDrop, false);
    fileInput.addEventListener('change', handleFileSelect, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    function handleFileSelect(e) {
        const files = e.target.files;
        handleFiles(files);
    }

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (validateFile(file)) {
                showFilePreview(file);
            }
        }
    }

    function validateFile(file) {
        // 检查文件类型和大小
        const validTypes = ['text/plain', 'application/fasta'];
        const maxSize = 5 * 1024 * 1024; // 5MB

        if (!validTypes.includes(file.type) && !file.name.endsWith('.fasta')) {
            alert('Please upload a FASTA or TXT file');
            return false;
        }

        if (file.size > maxSize) {
            alert('File size should not exceed 5MB');
            return false;
        }

        return true;
    }

    function showFilePreview(file) {
        fileName.textContent = file.name;
        fileSize.textContent = `(${formatFileSize(file.size)})`;
        dropZoneContent.style.display = 'none';
        filePreview.style.display = 'flex';
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // 移除文件
    removeFileBtn.addEventListener('click', function() {
        fileInput.value = '';
        dropZoneContent.style.display = 'block';
        filePreview.style.display = 'none';
    });

    // 表单提交处理
    const fileUploadForm = document.getElementById('fileUploadForm');
    const sequenceForm = document.getElementById('sequenceForm');

    fileUploadForm.addEventListener('submit', handleSubmit);
    sequenceForm.addEventListener('submit', handleSubmit);

    function handleSubmit(e) {
        e.preventDefault();
        // 这里添加提交逻辑
        console.log('Form submitted');
    }
}); 